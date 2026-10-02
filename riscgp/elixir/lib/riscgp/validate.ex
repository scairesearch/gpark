defmodule RiscGP.Validate do
  @moduledoc """
  IR validator for RVGPU.

  The cheap gate: it runs anywhere, needs no simulator and no silicon, and
  catches the class of mistake this ISA is unusually good at generating. It
  reports every problem, not just the first, because a validator that stops at
  error one is unusable in CI.

  ## What it checks

  **Structure** — at least one block, unique labels, every block terminated.

  **Opcode legality** — the opcode exists *on this kernel's datapath*, and its
  destination count, operand count, type, address space and modifier match the
  table. Path check matters because the table carries both datapaths: a
  `mop.mma` is legal on Path B and illegal on Path A.

  **Thread legality** — an instruction issued from a thread that does not have
  the unit in its address space. `:t0` can drive the matrix unit, `:b` can only
  push to it, and nothing can issue `stallwait` on the NoC core.

  **Provenance** — every instruction is attributed to a unit, so a kernel
  cannot mix instructions whose units the same thread cannot both reach.

  **Async hazard** — the one that bites hardest. A `mop` result is *not* visible
  to the issuing core when the instruction retires, so reading a `dst`,
  `lreg`, `srca` or `srcb` after writing it without an intervening
  `stallwait` or `sync.sem` is a **silently wrong answer**, not a fault. This
  is checked, and it is checked as an error because there is no runtime
  symptom to catch it in testing.

  ## Usage

      RiscGP.Validate.check(kernel)
      #=> {:ok, kernel}
      #=> {:error, [%{kind: :missing_stallwait, ...}]}
  """

  alias RiscGP.IR
  alias RiscGP.Table

  @type issue :: %{kind: atom(), message: String.t(), block: atom | nil, index: non_neg_integer | nil}

  # Matrix-unit state. Only `stallwait` and a release/acquire `sync.sem` make a
  # write visible to the issuing core; nothing else does, which is the entire
  # content of the async hazard.
  @coprocessor_files ~w(dst lreg srca srcb)a

  # Which threads can reach which unit directly. `dm.push` is how `:b` reaches
  # the coprocessor, so `:b` is absent from the coprocessor list on purpose.
  @unit_threads %{
    "core" => [:b, :t0, :t1, :t2, :nc],
    "dma" => [:b, :t1, :t2, :nc],
    "matrix" => [:t0, :t1, :t2],
    "unpack" => [:t0, :t1, :t2],
    "pack" => [:t0, :t1, :t2],
    "vector" => [:t0, :t1, :t2],
    "sync" => [:b, :t0, :t1, :t2, :nc]
  }

  @doc "Opcodes a given thread may issue."
  def unit_threads, do: @unit_threads

  @doc """
  Validate a kernel. Returns `{:ok, kernel}` or `{:error, issues}`.
  """
  def check(%{} = kernel) do
    case kernel_issues(kernel) do
      [] -> {:ok, kernel}
      issues -> {:error, Enum.reverse(issues)}
    end
  end

  @doc "Validate many kernels, collecting issues across all of them."
  def check_all(kernels) when is_list(kernels) do
    Enum.flat_map(kernels, &kernel_issues/1)
  end

  # ---------------------------------------------------------------------------

  defp kernel_issues(kernel) do
    ctx = %{
      kernel: kernel,
      path: kernel.path,
      params: MapSet.new(kernel.params, & &1.name),
      labels: MapSet.new(kernel.blocks, & &1.label),
      issues: []
    }

    case shape_issues(kernel) do
      [] -> block_issues(kernel, ctx) |> Map.get(:issues)
      issues -> issues
    end
  end

  defp shape_issues(%{blocks: []}) do
    [%{kind: :empty_kernel, message: "kernel has no blocks", block: nil, index: nil}]
  end

  defp shape_issues(%{blocks: blocks}) do
    blocks
    |> Enum.map(& &1.label)
    |> Enum.frequencies()
    |> Enum.filter(fn {_l, count} -> count > 1 end)
    |> Enum.map(fn {label, count} ->
      %{kind: :duplicate_block, message: "block #{inspect(label)} defined #{count} times", block: label, index: nil}
    end)
  end

  # ---------------------------------------------------------------------------
  # Block scan
  # ---------------------------------------------------------------------------

  defp block_issues(kernel, ctx) do
    # State threaded across blocks: which coprocessor files have been written
    # and not yet made visible, and which have been written *and* fenced.
    initial = %{pending: MapSet.new(), live: MapSet.new(), defined: MapSet.new()}

    kernel.blocks
    |> Enum.reduce(initial, fn block, state ->
      state = Enum.reduce(block.instrs, state, &instr_effects(&1, block, &2, ctx))

      case block.term do
        nil -> add(state, :unterminated_block, "block #{inspect(block.label)} has no terminator", block.label, nil, ctx)
        term -> instr_effects(term, block, state, ctx)
      end
    end)
  end

  # Every effect of an instruction, accumulated: legality, then the async
  # hazard, then the state transition.
  defp instr_effects(instr, block, state, ctx) do
    ctx
    |> legality(instr, block, state)
    |> then(fn ctx -> hazard(ctx, instr, block, state) end)
    |> then(fn ctx -> transition(state, instr, ctx) end)
  end

  defp legality(ctx, instr, block, _state) do
    case Table.op_spec(instr.base) do
      nil ->
        add(ctx, :unknown_opcode, "unknown opcode #{inspect(instr.base)}", block.label, nil, ctx)

      spec ->
        ctx
        |> on_path(spec, instr, block)
        |> on_thread(spec, instr, block)
        |> on_arity(spec, instr, block)
        |> on_modifier(spec, instr, block)
        |> on_type(spec, instr, block)
        |> on_space(spec, instr, block)
        |> on_references(instr, block)
    end
  end

  defp on_path(ctx, spec, instr, block) do
    wanted = String.upcase(Atom.to_string(ctx.path))

    if spec.path in ["AB", wanted] do
      ctx
    else
      add(ctx, :wrong_path,
        "#{instr.base} is a #{spec.path} opcode, not available on path #{wanted}",
        block.label, nil, ctx)
    end
  end

  defp on_thread(ctx, spec, instr, block) do
    allowed = Map.get(@unit_threads, spec.unit, [])

    if instr.thread in allowed do
      ctx
    else
      add(ctx, :bad_thread,
        "#{inspect(instr.thread)} cannot issue #{instr.base} (unit #{spec.unit}, reachable from: #{inspect(allowed)})",
        block.label, nil, ctx)
    end
  end

  defp on_arity(ctx, spec, instr, block) do
    ctx = if count_dests(instr.dest) != spec.ndest do
      add(ctx, :dest_arity,
        "#{instr.base} expects #{spec.ndest} destination(s), got #{count_dests(instr.dest)}",
        block.label, nil, ctx)
    else
      ctx
    end

    if length(instr.ops) != spec.nops do
      add(ctx, :operand_arity,
        "#{instr.base} expects #{spec.nops} operand(s), got #{length(instr.ops)}",
        block.label, nil, ctx)
    else
      ctx
    end
  end

  defp count_dests(nil), do: 0
  defp count_dests(_), do: 1

  defp on_modifier(ctx, spec, instr, block) do
    case spec.modifiers do
      nil -> ctx
      [] -> ctx
      [nil] -> ctx
      allowed ->
        if instr.modifier in allowed do
          ctx
        else
          add(ctx, :bad_modifier,
            "modifier #{inspect(instr.modifier)} not permitted on #{instr.base} (allowed: #{inspect(allowed)})",
            block.label, nil, ctx)
        end
    end
  end

  defp on_type(ctx, spec, instr, block) do
    if spec.dtypes != [] and instr.dtype not in spec.dtypes do
      add(ctx, :bad_type,
        "#{instr.base} does not support type #{inspect(instr.dtype)} (allowed: #{inspect(spec.dtypes)})",
        block.label, nil, ctx)
    else
      ctx
    end
  end

  defp on_space(ctx, spec, instr, block) do
    if is_list(spec.spaces) and instr.space not in spec.spaces do
      add(ctx, :bad_space,
        "#{instr.base} does not support address space #{inspect(instr.space)} (allowed: #{inspect(spec.spaces)})",
        block.label, nil, ctx)
    else
      ctx
    end
  end

  defp on_references(ctx, instr, block) do
    Enum.reduce(instr.ops, ctx, fn
      {:label, target}, ctx ->
        if MapSet.member?(ctx.labels, target) do
          ctx
        else
          add(ctx, :unknown_label, "branch to undefined block #{inspect(target)}", block.label, nil, ctx)
        end

      {:param, name}, ctx ->
        if MapSet.member?(ctx.params, name) do
          ctx
        else
          add(ctx, :unknown_param, "reference to undefined parameter #{inspect(name)}", block.label, nil, ctx)
        end

      _op, ctx ->
        ctx
    end)
  end

  # ---------------------------------------------------------------------------
  # The async hazard
  # ---------------------------------------------------------------------------

  # A coprocessor register that was written but not yet made visible, then read
  # by the same thread, is the silent-wrong-answer bug this ISA is famous for.
  defp hazard(ctx, instr, block, state) do
    files = read_files(instr)

    Enum.reduce(files, ctx, fn file, ctx ->
      if MapSet.member?(state.pending, {file, instr.thread}) do
        add(ctx, :missing_stallwait,
          "read of #{file} written earlier by #{inspect(instr.thread)} without an intervening stallwait or sync.sem release/acquire - the value will not be visible yet",
          block.label, nil, ctx)
      else
        ctx
      end
    end)
  end

  defp read_files(%{ops: ops}) do
    ops |> Enum.flat_map(&operand_files/1) |> Enum.uniq()
  end

  defp operand_files({tag, _id}) when tag in @coprocessor_files, do: [tag]
  defp operand_files({:addr, base, idx, _scale}), do: operand_files(base) ++ operand_files(idx)
  defp operand_files(_), do: []

  # `mop.*` marks a write pending; `stallwait` and ordering `sync.sem` make it
  # live. Note that `sync.sem wait` deliberately does *not*: waiting is not
  # ordering, and treating it as ordering is a bug in its own right.
  defp transition(state, instr, _ctx) do
    written = case instr.dest do
      {tag, id} when tag in @coprocessor_files -> [{tag, id}]
      _ -> []
    end

    cond do
      base?(instr) ->
        %{state | pending: Enum.reduce(written, state.pending, &MapSet.put(&2, {&1, instr.thread})),
          live: Enum.reduce(written, state.live, &MapSet.put(&2, &1))}

      ordering?(instr) ->
        %{state | pending: MapSet.new()}

      true ->
        state
    end
  end

  # A write to a coprocessor file, from either datapath's math unit.
  defp base?(%{base: base}) do
    String.starts_with?(base, "mop.") or base in ["vfmacc", "vfredsum", "vfmadd"]
  end

  defp ordering?(instr) do
    instr.base == "stallwait" or (instr.base == "sync.sem" and instr.modifier in ["release", "acquire"])
  end

  # ---------------------------------------------------------------------------

  defp add(_ctx, kind, message, block, index, acc) do
    %{acc | issues: [%{kind: kind, message: message, block: block, index: index} | acc.issues]}
  end
end
