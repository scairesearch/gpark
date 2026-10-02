defmodule Gpark.Validate do
  @moduledoc """
  IR type-checker and structural validator for `Gpark.IR`.

  This is the first of the four validation layers described in
  `docs/VALIDATION.md`. It runs anywhere — no GPU, no `ptxas`, no network — so it
  is the cheap gate that catches mistakes before a kernel ever reaches the
  remote NVIDIA host.

  It checks:

    * kernel structure: at least one block, unique labels, every block
      terminated, at most one `exit`
    * opcode legality: the opcode exists, its operand and destination counts
      match the typed table, and its type/space/modifier are permitted
    * register typing: a register id is never used at two different types, and
      predicate registers are only used as predicates
    * initialisation: every register is written before it is read. PTX does not
      zero-initialise registers, and a kernel that reads one is a bug that
      reproduces only under a different block schedule — exactly the kind of
      heisenbug this project exists to not ship
    * references: branch targets and parameters actually exist

  ## Usage

      Gpark.Validate.check(kernel)
      #=> {:ok, kernel}
      #=> {:error, [%{kind: :unknown_opcode, message: "...", ...}]}
  """

  alias Gpark.IR

  @type issue :: %{
          kind: atom(),
          message: String.t(),
          block: atom() | nil,
          index: non_neg_integer() | nil
        }

  @doc """
  Validate a kernel.

  Returns `{:ok, kernel}` or `{:error, issues}` with every problem found, not
  just the first. A compiler that reports one error per run is unusable in CI.
  """
  @spec check(map()) :: {:ok, map()} | {:error, [issue()]}
  def check(%{} = kernel), do: check(kernel, context(kernel))

  @doc "Validate a list of kernels at once, collecting issues across all of them."
  def check_all(kernels) when is_list(kernels) do
    case Enum.flat_map(kernels, fn k ->
           case check(k) do
             {:error, issues} -> issues
             {:ok, _} -> []
           end
         end) do
      [] -> :ok
      issues -> {:error, issues}
    end
  end

  defp context(kernel) do
    params = MapSet.new(kernel.params, & &1.name)
    labels = MapSet.new(kernel.blocks, & &1.label)

    %{
      params: params,
      labels: labels,
      written: MapSet.new(),
      # reg id -> type
      typed: %{},
      exits: 0,
      issues: []
    }
  end

  # ---------------------------------------------------------------------------

  defp check(kernel, ctx) do
    ctx = validate_shape(kernel, ctx)
    ctx = if ctx.issues == [], do: scan_blocks(kernel, ctx), else: ctx

    if ctx.issues == [] do
      {:ok, kernel}
    else
      {:error, Enum.reverse(ctx.issues)}
    end
  end

  defp validate_shape(%{blocks: []}, ctx) do
    issue(ctx, :empty_kernel, "kernel has no blocks")
  end

  defp validate_shape(%{blocks: blocks}, ctx) do
    blocks
    |> Enum.map(& &1.label)
    |> Enum.frequencies()
    |> Enum.filter(fn {_label, count} -> count > 1 end)
    |> Enum.reduce(ctx, fn {label, count}, ctx ->
      issue(ctx, :duplicate_block, "block #{inspect(label)} defined #{count} times")
    end)
  end

  # ---------------------------------------------------------------------------
  # Block scan
  # ---------------------------------------------------------------------------

  defp scan_blocks(kernel, ctx) do
    Enum.reduce(kernel.blocks, ctx, fn block, ctx ->
      # The terminator is a real instruction and must be checked like any other.
      # Skipping it once meant a single-instruction block — which puts everything
      # in `term` — validated nothing at all.
      instrs = block.instrs ++ List.wrap(block.term)

      ctx =
        Enum.with_index(instrs)
        |> Enum.reduce(ctx, fn {instr, index}, ctx ->
          check_instruction(instr, block, index, ctx)
        end)

      ctx =
        if block.term do
          note_exit(block.term, ctx)
        else
          issue(
            ctx,
            :unterminated_block,
            "block #{inspect(block.label)} has no terminator",
            block.label,
            length(block.instrs) - 1
          )
        end

      ctx
    end)
  end

  # Deliberately narrow: only an explicit `exit` is a hazard. A kernel with an
  # early-return block and a `done` block legitimately contains several `ret`s and
  # PTX is perfectly happy with that, so counting returns would fire on correct
  # code. Multiple `exit`s, by contrast, really do mean divergent threads are being
  # torn down twice.
  defp note_exit(%{base: "exit"}, ctx) do
    issue(ctx, :multiple_exits, "kernel contains more than one exit")
  end

  defp note_exit(_, ctx), do: ctx

  # ---------------------------------------------------------------------------
  # Per-instruction checks
  # ---------------------------------------------------------------------------

  defp check_instruction(%{base: base}, block, index, ctx) do
    case IR.op_spec(base) do
      nil ->
        issue(ctx, :unknown_opcode, "unknown opcode #{inspect(base)}", block.label, index)

      spec ->
        ctx
        |> check_arity(spec, block, index)
        |> check_modifiers(spec, block, index)
        |> check_types(spec, block, index)
        |> check_references(block, index)
        |> check_typing(block, index)
        |> check_initialisation(block, index)
        |> record_written(block, index)
    end
  end

  defp check_arity(ctx, spec, block, index) do
    %{dest: dest, ops: ops, base: base} = instruction_at(block, index)
    actual_dests = if dest, do: 1, else: 0

    ctx =
      if actual_dests != spec.ndest do
        issue(
          ctx,
          :dest_arity,
          "#{base} expects #{spec.ndest} destination(s), got #{actual_dests}",
          block.label,
          index
        )
      else
        ctx
      end

    if length(ops) != spec.nops do
      issue(
        ctx,
        :operand_arity,
        "#{base} expects #{spec.nops} operand(s), got #{length(ops)}",
        block.label,
        index
      )
    else
      ctx
    end
  end

  defp check_modifiers(ctx, spec, block, index) do
    # `ld`/`st` carry a modifier that is deliberately nil-able.
    case spec do
      %{modifiers: allowed} when is_list(allowed) and allowed != [] ->
        mod = modifier_of(block, index)

        if allowed == [nil] or mod in allowed do
          ctx
        else
          issue(
            ctx,
            :bad_modifier,
            "modifier #{inspect(mod)} not permitted here (allowed: #{inspect(allowed)})",
            block.label,
            index
          )
        end

      _ ->
        ctx
    end
  end

  # The opcode's `parts` list is the authority on whether a given instruction
  # even has a space or modifier field.
  defp modifier_of(block, index) do
    block.instrs
    |> Enum.concat(List.wrap(block.term))
    |> Enum.at(index)
    |> Map.get(:modifier)
  end

  defp check_types(ctx, spec, block, index) do
    instr = instruction_at(block, index)

    # An empty dtype list means the opcode has no type suffix at all (`bra`,
    # `ret`, `bar`, …), so there is nothing to check.
    ctx =
      if spec.dtype != [] and instr.dtype not in spec.dtype do
        issue(
          ctx,
          :bad_type,
          "#{instr.base} does not support type #{inspect(instr.dtype)} (allowed: #{inspect(spec.dtype)})",
          block.label,
          index
        )
      else
        ctx
      end

    ctx =
      if is_list(spec.spaces) and instr.space not in spec.spaces do
        issue(
          ctx,
          :bad_space,
          "#{instr.base} does not support address space #{inspect(instr.space)}",
          block.label,
          index
        )
      else
        ctx
      end

    ctx = check_operand_types(ctx, spec, instr, block, index)

    if is_list(spec.srcs) and instr.srctype not in spec.srcs do
      issue(
        ctx,
        :bad_source_type,
        "#{instr.base} does not convert from #{inspect(instr.srctype)}",
        block.label,
        index
      )
    else
      ctx
    end
  end

  # The `otypes` column was documented in `Gpark.Ops` but never enforced, so the
  # table quietly disagreed with the checker. `:same` means every typed register
  # operand must match the opcode's own type — this is what catches `add.u32`
  # quietly operating on an f32 register.
  defp check_operand_types(ctx, spec, instr, block, index) do
    allowed =
      case spec.otypes do
        :same -> spec.dtype
        :any -> nil
        :sreg -> nil
        list -> list
      end

    if allowed do
      Enum.reduce(instr.ops, ctx, fn
        {:reg, type, _id}, ctx ->
          if type in allowed do
            ctx
          else
            issue(
              ctx,
              :operand_type_mismatch,
              "#{instr.base}.#{inspect(instr.dtype)} does not accept a #{inspect(type)} operand (allowed: #{inspect(allowed)})",
              block.label,
              index
            )
          end

        _op, ctx ->
          ctx
      end)
    else
      ctx
    end
  end

  # ---------------------------------------------------------------------------
  # References: branches and parameters must resolve
  # ---------------------------------------------------------------------------

  defp check_references(ctx, block, index) do
    instr = instruction_at(block, index)

    Enum.reduce(instr.ops, ctx, fn
      {:label, target}, ctx ->
        if MapSet.member?(ctx.labels, target) do
          ctx
        else
          # Note: `issue/5` returns an updated ctx, so it must not be wrapped in
          # push/2 — that would append the entire context to the issues list.
          ctx
          |> issue(
            :unknown_label,
            "branch to undefined block #{inspect(target)}",
            block.label,
            index
          )
          |> Map.update!(:labels, &MapSet.put(&1, target))
        end

      {:param, name}, ctx ->
        if MapSet.member?(ctx.params, name) do
          ctx
        else
          issue(
            ctx,
            :unknown_param,
            "reference to undefined parameter #{inspect(name)}",
            block.label,
            index
          )
        end

      {:sreg, name}, ctx ->
        if Map.has_key?(IR.sreg_names(), name) do
          ctx
        else
          issue(
            ctx,
            :unknown_sreg,
            "unknown special register #{inspect(name)}",
            block.label,
            index
          )
        end

      _op, ctx ->
        ctx
    end)
  end

  # ---------------------------------------------------------------------------
  # Register typing: one id, one type
  # ---------------------------------------------------------------------------

  defp check_typing(ctx, block, index) do
    instr = instruction_at(block, index)

    IR.instr_regs(instr)
    |> Enum.reduce(ctx, fn {type, class, id}, ctx ->
      key = {class, id}

      case Map.get(ctx.typed, key) do
        nil ->
          Map.put(ctx, :typed, Map.put(ctx.typed, key, type))

        ^type ->
          ctx

        previous ->
          issue(
            ctx,
            :register_type_conflict,
            "%#{class}#{id} used as #{type} here but #{previous} earlier",
            block.label,
            index
          )
      end
    end)
  end

  # ---------------------------------------------------------------------------
  # Initialisation: PTX does not zero registers
  # ---------------------------------------------------------------------------

  defp check_initialisation(ctx, block, index) do
    instr = instruction_at(block, index)

    # Only `written` suppresses this. Checking `typed` here would be circular:
    # `check_typing/3` runs first and records every register it *sees*, so a
    # register that was only ever read would look initialised.
    read_ids(instr)
    |> Enum.reduce(ctx, fn {class, id}, ctx ->
      if MapSet.member?(ctx.written, {class, id}) do
        ctx
      else
        issue(
          ctx,
          :uninitialised_register,
          "read of never-written register %#{class}#{id}",
          block.label,
          index
        )
      end
    end)
  end

  # Every register this instruction reads, as `{class, id}`. For the two-operand
  # memory ops (`st`, `red`) the first operand is an address rather than a value,
  # but it is still a register that has to have been written.
  defp read_ids(%{ops: ops, pred: pred}) do
    reads =
      Enum.flat_map(ops, fn
        {:reg, type, id} -> [{IR.reg_class(type), id}]
        {:pred, id} -> [{:p, id}]
        {:addr, base, idx, _scale} -> operand_ids(base) ++ operand_ids(idx)
        _ -> []
      end)

    reads ++ guard_id(pred)
  end

  defp guard_id(nil), do: []
  defp guard_id({:pred, id}), do: [{:p, id}]

  defp operand_ids({:reg, type, id}), do: [{IR.reg_class(type), id}]
  defp operand_ids(_), do: []

  defp record_written(ctx, block, index) do
    instr = instruction_at(block, index)

    dest_ids =
      case instr.dest do
        nil -> []
        {:reg, type, id} -> [{IR.reg_class(type), id}]
        {:pred, id} -> [{:p, id}]
      end

    Map.update!(ctx, :written, &MapSet.union(&1, MapSet.new(dest_ids)))
  end

  # ---------------------------------------------------------------------------

  defp instruction_at(block, index) do
    block.instrs |> Enum.concat(List.wrap(block.term)) |> Enum.at(index)
  end

  defp issue(ctx, kind, message, block \\ nil, index \\ nil) do
    %{ctx | issues: [%{kind: kind, message: message, block: block, index: index} | ctx.issues]}
  end
end
