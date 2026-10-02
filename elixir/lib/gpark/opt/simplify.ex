defmodule Gpark.Opt.Simplify do
  @moduledoc """
  Simplification over `Gpark.IR`, run to a fixpoint.

  This is where gpark adopts one specific thing from Taichi: `full_simplify` loops
  until no rule fires rather than running a fixed number of passes. A bounded pass
  count makes optimisation quality depend on a tuning constant nobody understands,
  and it fails silently -- the output is still correct, just less simplified, so
  nothing tells you to raise the bound. A fixpoint either converges or a rule is
  wrong, and this module raises on non-convergence instead of quietly returning a
  half-simplified kernel.

  ## What is in v0.1

  Two rules, both chosen because they are hard to get subtly wrong:

    * `drop_unreachable_blocks` -- blocks no path can reach.
    * `drop_dead_instructions` -- side-effect-free instructions whose destination is
      never read.

  ## Loads are deliberately not dead code

  A dead `ld.global` is removable in any normal compiler, and this module does not
  remove it. A load whose result is unused can still fault, and removing it would
  paper over an out-of-bounds access.

  That is not hypothetical. `unpack_u4_f32` once emitted its bounds guard *after*
  the work it guarded, so every out-of-range lane performed a 32-byte out-of-bounds
  write. A dead-load rule would have deleted the evidence and left the kernel looking
  clean. gpark keeps loads: a rule that hides faults is worse than a missed
  optimisation.

  ## Limits

  Liveness is kernel-wide, not per-block: an instruction goes only when its
  destination is never read *anywhere* in the kernel. That errs in the safe
  direction -- it cannot remove a write a later block depends on -- but it also
  cannot remove the first of two writes to a register whose second write is live.
  Catching that needs a real CFG and per-block liveness, deferred until a kernel
  needs it.

  `simplify/2` reports per-rule counts so a rule that never fires is visible. A pass
  where every count is zero is either running on already-minimal code or is broken,
  and the counts are what tell those apart.
  """

  # Opcodes whose only effect is to write their destination. Removing one whose
  # result is unused cannot change behaviour.
  #
  # Listed exhaustively on purpose. A prefix rule like `String.starts_with?(base,
  # "st")` is silently wrong the day someone adds an opcode: anything unrecognised
  # falls through as "pure" and the pass starts deleting instructions that write
  # memory. `test "every opcode is classified"` fails CI instead.
  @pure ~w(
    abs add and brev clz cvt cvta div fma mad mad.hi mad.lo max min mov mul mul.wide
    neg nop not or popc rcp rem rsqrt s2r selp setp shl shr slct sqrt sub xor
  )

  # Opcodes that do something beyond writing their destination.
  #
  #   memory writes        st
  #   memory reads         ld prefetch        (kept on purpose: see moduledoc)
  #   control flow         bra brx ret exit
  #   synchronisation      bar shfl vote activemask
  #   atomics              atom red
  #   calls                call.uni
  @impure ~w(
    activemask atom bar bra brx call.uni exit ld prefetch red ret shfl st vote
  )

  # A fixpoint on a kernel this size converges in a handful of iterations. This is a
  # guard against a rule that oscillates, not a tuning parameter: exceeding it raises
  # rather than returning unsimplified output.
  @max_iterations 32

  @rules [:drop_unreachable_blocks, :drop_dead_instructions]

  @doc "Simplify `kernel` to a fixpoint, returning the kernel."
  @spec simplify(map()) :: map()
  def simplify(kernel), do: elem(simplify(kernel, []), 0)

  @doc """
  Simplify `kernel` to a fixpoint, reporting what happened.

  Returns `{kernel, stats}`, where stats is
  `%{iterations: non_neg_integer, rules: %{atom => non_neg_integer}}`.

  Every rule appears in `rules`, including at zero, so a rule that never fires is
  distinguishable from a rule that does not exist.
  """
  @spec simplify(map(), keyword()) :: {map(), map()}
  def simplify(kernel, _opts) do
    iterate(kernel, 0, Map.new(@rules, &{&1, 0}))
  end

  @doc "Whether `simplify/1` would change `kernel`."
  @spec changed?(map()) :: boolean()
  def changed?(kernel), do: simplify(kernel) != kernel

  @doc """
  Side-effect classification for an opcode name.

  Returns `:pure`, `:impure` or `:unknown`. `:unknown` is a real answer, not a
  fallback: an opcode nobody has classified has to be treated as impure, because
  assuming purity is how a pass starts deleting instructions that write memory.
  """
  @spec classify(binary()) :: :pure | :impure | :unknown
  def classify(base) when is_binary(base) do
    cond do
      base in @pure -> :pure
      base in @impure -> :impure
      true -> :unknown
    end
  end

  def classify(_), do: :unknown

  @doc "True when an opcode may be removed if its result is unused."
  @spec pure?(map() | binary()) :: boolean()
  def pure?(base) when is_binary(base), do: classify(base) == :pure
  def pure?(%{base: base}), do: pure?(base)

  @doc "The pure opcode list."
  def pure_ops, do: @pure

  @doc "The impure opcode list."
  def impure_ops, do: @impure

  # ---------------------------------------------------------------------------
  # Fixpoint driver
  # ---------------------------------------------------------------------------

  defp iterate(kernel, iteration, acc) do
    if iteration > @max_iterations do
      raise ArgumentError,
            "simplification of #{kernel.name} did not converge in #{@max_iterations} " <>
              "iterations; a rule is probably oscillating. Returning half-simplified " <>
              "output would hide that, so this raises instead."
    end

    {next, fired} = pass(kernel)

    if next == kernel do
      {next, %{iterations: iteration, rules: acc}}
    else
      iterate(next, iteration + 1, merge_counts(acc, fired))
    end
  end

  defp pass(kernel) do
    {blocks, unreachable} = drop_unreachable_blocks(kernel)
    {kept, dead} = drop_dead_instructions(blocks, kernel)

    {%{kernel | blocks: kept},
     %{drop_unreachable_blocks: unreachable, drop_dead_instructions: dead}}
  end

  defp merge_counts(acc, new) do
    Enum.reduce(new, acc, fn {k, v}, m -> Map.update(m, k, v, &(&1 + v)) end)
  end

  # ---------------------------------------------------------------------------
  # Rule: unreachable blocks
  # ---------------------------------------------------------------------------

  defp drop_unreachable_blocks(kernel) do
    reachable = reachable_from(kernel)

    kept = Enum.filter(kernel.blocks, &MapSet.member?(reachable, &1.label))
    {kept, length(kernel.blocks) - length(kept)}
  end

  defp reachable_from(kernel) do
    case kernel.blocks do
      [] -> MapSet.new()
      [root | _] -> walk(MapSet.new([root.label]), kernel)
    end
  end

  # Classic worklist-by-repetition: cheap at this size, and obvious to verify. The
  # only cost is repeating over all blocks, which is nothing for a kernel with two.
  defp walk(set, kernel) do
    expanded =
      Enum.reduce(kernel.blocks, set, fn block, acc ->
        if MapSet.member?(acc, block.label) do
          Enum.reduce(successors(block, kernel), acc, &MapSet.put(&2, &1))
        else
          acc
        end
      end)

    if MapSet.equal?(expanded, set), do: set, else: walk(expanded, kernel)
  end

  defp successors(block, kernel) do
    branch_targets(block) ++ fallthrough(block, kernel)
  end

  defp branch_targets(block) do
    (block.instrs ++ List.wrap(block.term))
    |> Enum.flat_map(& &1.ops)
    |> Enum.filter(&match?({:label, _}, &1))
    |> Enum.map(fn {:label, label} -> label end)
  end

  # A predicated branch falls through; an unpredicated one does not. Honouring the
  # unconditional case is what lets a block sitting between `bra :done` and `:done`
  # be recognised as unreachable.
  #
  # Anything else that is not `ret`/`exit` is treated as falling through, `brx`
  # included, since its predicate decides at runtime. Over-approximating `brx` only
  # ever keeps more blocks, and keeping a block that turned out to be reachable is
  # the safe direction for a deletion pass.
  defp fallthrough(block, kernel) do
    if falls_through?(block.term) do
      case next_block(kernel, block.label) do
        nil -> []
        next -> [next.label]
      end
    else
      []
    end
  end

  defp next_block(kernel, label) do
    index = Enum.find_index(kernel.blocks, &(&1.label == label))

    if is_nil(index) do
      nil
    else
      Enum.at(kernel.blocks, index + 1)
    end
  end

  defp falls_through?(nil), do: true
  defp falls_through?(%{base: base}) when base in ["ret", "exit"], do: false
  defp falls_through?(%{base: "bra", pred: nil}), do: false
  defp falls_through?(%{}), do: true
  defp falls_through?(_), do: true

  # ---------------------------------------------------------------------------
  # Rule: dead instructions
  # ---------------------------------------------------------------------------

  defp drop_dead_instructions(blocks, kernel) do
    live = live_registers(kernel)

    {kept, removed} =
      Enum.reduce(blocks, {[], 0}, fn block, {acc, n} ->
        {instrs, dropped} =
          Enum.reduce(block.instrs, {[], 0}, fn instr, {keep, d} ->
            if removable?(instr, live) do
              {keep, d + 1}
            else
              {[instr | keep], d}
            end
          end)

        {acc ++ [%{block | instrs: Enum.reverse(instrs)}], n + dropped}
      end)

    {kept, removed}
  end

  defp removable?(%{base: base, dest: dest}, live) do
    case dest_id(dest) do
      nil -> false
      id -> pure?(base) and not MapSet.member?(live, id)
    end
  end

  defp removable?(_, _), do: false

  # Registers read somewhere in the kernel, keyed by the same {type, class, id} triple
  # the IR uses. gpark reuses numeric ids across types -- u32 1 and u64 1 are
  # different registers -- so keying on a bare id would conflate them and delete
  # instructions that are very much alive.
  defp live_registers(kernel) do
    for block <- kernel.blocks,
        instr <- block.instrs ++ List.wrap(block.term),
        key = {_, _, _} <- read_regs(instr),
        into: MapSet.new() do
      key
    end
  end

  defp read_regs(%{ops: ops, pred: pred}) do
    Enum.flat_map(ops, &operand_reads/1) ++ predicate_reads(pred)
  end

  defp read_regs(_), do: []

  defp operand_reads({:reg, type, id}), do: [{type, :r, id}]
  defp operand_reads({:pred, id}), do: [{:pred, :p, id}]
  defp operand_reads({:addr, base, idx, _scale}), do: operand_reads(base) ++ operand_reads(idx)
  defp operand_reads(_), do: []

  defp predicate_reads({:pred, id}), do: [{:pred, :p, id}]
  defp predicate_reads(_), do: []

  defp dest_id({:reg, type, id}), do: {type, :r, id}
  defp dest_id({:pred, id}), do: {:pred, :p, id}
  defp dest_id(_), do: nil
end
