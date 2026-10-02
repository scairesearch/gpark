defmodule RiscGP.Cycle do
  @moduledoc """
  Static cycle-counting model for RVGPU.

  This is the P0 deliverable and the input to validation layer **L2**: it walks
  a kernel in program order and produces a cycle count, *derived only from the
  opcode table and the spec*. It shares no scheduling code with a future
  Verilator model, so when the two disagree, one of them is wrong — and that
  disagreement is the bug-finding mechanism. A single "obviously correct"
  model checked against itself is worth nothing.

  ## What is modelled

  - **One in-order issue port per core thread**, one instruction per cycle.
  - **Per-unit throughput.** A unit accepts a new instruction every
    `throughput` cycles. This is what makes `mop.mma` (throughput 32) scarce
    and `nop` (throughput 1) free.
  - **Latency before visibility.** A result becomes visible to its issuing
    thread at `issue + latency`, never at retire. On this ISA those differ, and
    the difference is the entire async hazard.
  - **Coprocessor reads wait for the write to complete**, so a missing
    `stallwait` shows up here as a real dependency rather than as a number that
    looks fine.
  - **`stallwait` waits for the named unit to drain**, which is what makes the
    Three-thread pattern measurable rather than decorative.
  - **Inter-thread sharing.** A file written by `:t0` and read by `:t1` is a
    cross-thread dependency, so each thread is scheduled independently and the
    cycle count is the makespan of all five.

  ## What is NOT modelled, and therefore why the numbers are estimates

  The table has `estimated` semantics throughout. Specifically this model does
  **not** account for:

  - **MOP expanders.** The spec (§4) says one incoming instruction can expand
    into many outgoing ones. Latency here is the *unexpanded* cost, so any
    kernel leaning on expansion is optimistic by an unknown factor. This is the
    single largest known gap.
  - **NoC contention.** `dma.mcast` and `dma.issue` are costed as one cycle of
    the DMA engine regardless of the number of tiles involved. Real multicast
    cost scales with the multicast rectangle, not with the instruction.
  - **SRAM bank conflicts.** A `lw` is costed at its sustained rate, not at a
    worst case that depends on which bank another thread is using.
  - **Register read-port conflicts**, which the spec notes are already
    asymmetric on Blackhole for `fmadd` versus `vfmacc`.

  Every figure this module produces is therefore a lower bound, and the P0 study
  is required to label them `estimate` rather than `measured`. Do not quote one
  of these numbers as a performance claim; see `docs/claims.md`.
  """

  alias RiscGP.IR
  alias RiscGP.Table

  @type result :: %{
          cycles: non_neg_integer(),
          threads: map(),
          unit_busy: map(),
          critical: [map()],
          caveats: [String.t()]
  }

  @doc """
  Count cycles for a kernel.

      RiscGP.Cycle.count(kernel)
      #=> %{cycles: 214, unit_busy: %{"matrix" => 128}, ...}
  """
  def count(%{} = kernel) do
    starts = schedule_per_thread(kernel)

    issue_cycles =
      kernel
      |> IR.all_instrs()
      |> Enum.map(fn instr -> Map.fetch!(starts, instr_key(instr)) end)
      |> Enum.map(fn {instr, cycle} -> %{instr: instr, cycle: cycle, unit: Table.unit(instr.base)} end)

    complete = Map.new(issue_cycles, fn e -> {e.cycle + Table.latency(e.instr.base), e.unit} end)

    %{
      cycles: max_cycles(issue_cycles),
      threads: Map.new(IR.threads(), &{&1, thread_cycles(kernel, starts, &1)}),
      unit_busy: unit_busy(issue_cycles, complete),
      critical: critical_path(issue_cycles),
      caveats: caveats()
    }
  end

  @doc "The model's known limitations, for attaching to any number it produces."
  def caveats do
    [
      "MOP expanders not modelled: latency is the unexpanded cost, so expander-using kernels are optimistic.",
      "NoC contention not modelled: dma.issue and dma.mcast cost one cycle regardless of tile count.",
      "SRAM bank conflicts not modelled: loads are costed at the sustained rate, not worst case.",
      "Register read-port conflicts not modelled.",
      "All figures are lower bounds and must be labelled estimate, not measured."
    ]
  end

  # ---------------------------------------------------------------------------
  # Per-thread list scheduling
  # ---------------------------------------------------------------------------

  # Each thread is scheduled independently through its own program order, which
  # is what makes cross-thread file dependencies real: a file written by one
  # thread simply is not written yet as far as the other is concerned, so the
  # reader's issue cycle is pushed out and the makespan grows.
  defp schedule_per_thread(kernel) do
    kernel.blocks
    |> Enum.group_by(& &1.thread)
    |> Enum.flat_map(fn {thread, blocks} ->
      {scheduled, _state} = schedule_stream(thread, blocks, initial_state())
      scheduled
    end)
    |> Map.new(fn {instr, cycle} -> {instr_key(instr), cycle} end)
  end

  defp initial_state, do: %{cursor: 0, unit_ready: %{}, written: %{}}

  defp schedule_stream(_thread, [], state), do: {[], state}

  defp schedule_stream(thread, [block | rest], state) do
    {acc, state} = Enum.map_reduce(block.instrs, state, &schedule_instr/2)

    {acc, state} =
      case block.term do
        nil ->
          {acc, state}

        term ->
          {scheduled, state} = schedule_instr(term, state)
          {acc ++ scheduled, state}
      end

    {more, state} = schedule_stream(thread, rest, state)
    {acc ++ more, state}
  end

  defp schedule_instr(instr, state) do
    unit = Table.unit(instr.base)
    ready = max(Map.get(state.unit_ready, unit, 0), dependencies_ready(instr, state))
    issued = max(state.cursor, ready)

    # `stallwait` blocks the thread until the named unit is idle, which is the
    # mechanism that makes a coprocessor result actually readable. The integer
    # operands name the unit to wait on.
    waited =
      if instr.base == "stallwait" do
        instr.ops
        |> Enum.filter(&is_integer/1)
        |> Enum.map(&Map.get(state.unit_ready, &1, 0))
        |> Enum.max(fn -> 0 end)
      else
        0
      end

    next = %{
      cursor: max(issued + 1, waited),
      unit_ready: Map.put(state.unit_ready, unit, issued + Table.throughput(instr.base)),
      written: record_written(instr, issued, state.written)
    }

    [{instr, issued}, next]
  end

  # When can this instruction's operands be read? A coprocessor file is only
  # written-and-complete once its latency has elapsed from the write's issue.
  defp dependencies_ready(instr, state) do
    instr.ops
    |> Enum.flat_map(&operand_files/1)
    |> Enum.map(fn {file, id} -> Map.get(state.written, {file, id}, 0) end)
    |> Enum.max(fn -> 0 end)
  end

  defp record_written(instr, issued, written) do
    case instr.dest do
      {tag, id} when tag in ~w(dst lreg srca srcb) ->
        Map.put(written, {tag, id}, issued + Table.latency(instr.base))

      _ ->
        written
    end
  end

  defp operand_files({tag, id}) when tag in ~w(dst lreg srca srcb), do: [{tag, id}]
  defp operand_files({:addr, base, idx, _scale}), do: operand_files(base) ++ operand_files(idx)
  defp operand_files(_), do: []

  # ---------------------------------------------------------------------------
  # Reporting
  # ---------------------------------------------------------------------------

  defp max_cycles([]), do: 0
  defp max_cycles(instrs), do: Enum.max(Enum.map(instrs, & &1.cycle))

  defp thread_cycles(kernel, starts, thread) do
    kernel
    |> IR.all_instrs()
    |> Enum.filter(&(&1.thread == thread))
    |> Enum.map(&Map.fetch!(starts, instr_key(&1)))
    |> case do
      [] -> 0
      cycles -> Enum.max(cycles) + 1
    end
  end

  # "Busy" is the span from first issue to last completion for a unit, which is
  # the number that tells you whether a kernel is matrix-bound or memory-bound.
  defp unit_busy(issue_cycles, complete) do
    units = issue_cycles |> Enum.map(& &1.unit) |> Enum.uniq()

    Map.new(units, fn unit ->
      first = issue_cycles |> Enum.filter(&(&1.unit == unit)) |> Enum.map(& &1.cycle) |> Enum.min()
      last = complete |> Enum.filter(fn {_, u} -> u == unit end) |> Enum.map(&elem(&1, 0)) |> Enum.max()
      {unit, last - first}
    end)
  end

  # The instructions that defined the makespan, in issue order. Read with the
  # unit label: if `matrix` dominates, the kernel is arithmetic-bound and the
  # core is idle anyway.
  defp critical_path(issue_cycles) do
    issue_cycles
    |> Enum.filter(fn e -> e.cycle + Table.latency(e.instr.base) == max_completion(issue_cycles) end)
    |> Enum.sort_by(& &1.cycle)
  end

  defp max_completion(instrs) do
    Enum.max(Enum.map(instrs, &(&1.cycle + Table.latency(&1.instr.base))))
  end

  defp instr_key(instr), do: {instr.thread, instr.base, instr.dest, instr.ops, instr.modifier, instr.space, instr.dtype}
end
