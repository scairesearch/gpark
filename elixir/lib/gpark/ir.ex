defmodule Gpark.IR do
  @moduledoc """
  Kernel IR for gpark.

  gpark targets hand-written PTX, so this IR is deliberately *not* SSA-like and
  does not hide machine state. Register identifiers are assigned explicitly by
  the caller, exactly as an assembly programmer would. That is the whole point:
  when a kernel is memory-bound, register pressure and scheduling are the levers,
  and an IR that quietly re-allocates them for you is working against you.

  Everything is a plain map so the IR round-trips through JSON. That is what lets
  `corpus/` hold portable kernel specs that both the Elixir and Python
  implementations read.

  ## Shapes

      kernel  %{name: atom, target: string, ptx_version: string, params: [param],
                shared: non_neg_integer, maxntid: tuple | nil, blocks: [block]}

      block   %{label: atom, instrs: [instr], term: instr | nil}

      instr   %{base: string, space: atom | nil, modifier: string | nil,
                dtype: atom, dest: dest | nil, ops: [operand], pred: pred | nil}

  Operands and destinations are tagged tuples:

      {:reg, type, id}    register, e.g. `{:reg, :f32, 1}`
      {:pred, id}         predicate register
      {:imm, value}       integer immediate
      {:immf, type, f}    float immediate
      {:param, name}      kernel parameter
      {:sreg, name}       special register, e.g. `{:sreg, :ctaid_x}`
      {:label, atom}      branch target
  """

  # ---------------------------------------------------------------------------
  # Types
  # ---------------------------------------------------------------------------

  # gpark's type vocabulary is owned by `Gpark.Type`, which models container
  # width and element format separately so that 2/4/8/16-bit types exist without
  # pretending PTX has 4-bit arithmetic. This list is the subset usable as a
  # direct operand type.
  @basic_types Gpark.Type.native_names()

  @doc "All PTX basic types gpark v0.1 models."
  def basic_types, do: @basic_types

  @doc "Delegate to `Gpark.Type.width/1`."
  defdelegate width(type), to: Gpark.Type

  @doc "Delegate to `Gpark.Type.kind/1`."
  defdelegate kind(type), to: Gpark.Type

  defdelegate float_type?(type), to: Gpark.Type, as: :float?
  defdelegate signed_type?(type), to: Gpark.Type, as: :int?
  defdelegate unsigned_type?(type), to: Gpark.Type, as: :int?
  def predicate_type?(type), do: type == :pred

  @doc """
  Register class for a type: `:r` (32-bit), `:rd` (64-bit), `:f` (float 32),
  `:fd` (float 64), `:p` (predicate).

  Note this is the *PTX register* class, which is deliberately coarser than the
  type, and narrower formats fold in where the ISA requires it:

    * `.s32`, `.u32`, `.b32` all live in `%r`; `.f32` has its own `%f` bank.
    * `.b1`/`.b2`/`.b4`/`.b8` have no register of their own width and are
      addressed through a 32-bit `%r`.
    * fp8 (`.e4m3`, `.e5m2`) and fp4/fp6 (`.e2m1`, `.e2m3`, `.e3m2`, `.e8m0`)
      are *converted from and to* `.b16`, so they live in `%r` too.
    * `.bf16` is a first-class arithmetic format with its own `%f` bank, unlike
      the fp8 family.

  This table is the reason gpark can offer 2/4/8-bit numerics without emitting
  PTX that `ptxas` rejects.
  """
  def reg_class(type)
  def reg_class(type) when is_atom(type) do
    cond do
      type in [:s64, :u64, :b64] -> :rd
      type in [:f32, :f16, :bf16] -> :f
      type == :f64 -> :fd
      type == :pred -> :p
      Gpark.Type.native?(type) -> :r
      true -> nil
    end
  end

  def reg_class(%Gpark.Type.Packed{container: container}), do: reg_class(container)
  def reg_class(_), do: nil

  # ---------------------------------------------------------------------------
  # Address spaces
  # ---------------------------------------------------------------------------

  @spaces ~w(global shared local const param)a

  @doc "Address spaces gpark v0.1 models."
  def spaces, do: @spaces

  # ---------------------------------------------------------------------------
  # Special registers
  # ---------------------------------------------------------------------------

  @sreg_names %{
    tid_x: "tid.x", tid_y: "tid.y", tid_z: "tid.z",
    ctaid_x: "ctaid.x", ctaid_y: "ctaid.y", ctaid_z: "ctaid.z",
    ntid_x: "ntid.x", ntid_y: "ntid.y", ntid_z: "ntid.z",
    nctaid_x: "nctaid.x", nctaid_y: "nctaid.y", nctaid_z: "nctaid.z",
    laneid: "laneid", warpid: "warpid", nwarpid: "nwarpid", griddep: "griddep"
  }

  @doc "Special registers, keyed to their PTX spelling."
  def sreg_names, do: @sreg_names

  # ---------------------------------------------------------------------------
  # Opcode table
  # ---------------------------------------------------------------------------

  # The opcode table lives in `Gpark.Ops`. It cannot be a module attribute here:
  # a module attribute cannot call a local function, so `@ops %{...}` built from
  # helper calls does not compile. One indirection, and the table becomes
  # importable data rather than something the IR module both owns and needs.

  @doc "The typed opcode table. See `docs/PTX-SUBSET.md` for what this covers."
  defdelegate ops, to: Gpark.Ops, as: :table

  @doc "Look up an opcode entry by base name, or nil."
  defdelegate op_spec(base), to: Gpark.Ops, as: :fetch

  @doc "Default target architecture when a caller does not pick one."
  def default_target, do: "sm_80"

  @doc "Default `.version` emitted when a caller does not pick one."
  def default_ptx_version, do: "8.7"

  # ---------------------------------------------------------------------------
  # Constructors
  # ---------------------------------------------------------------------------

  @doc "A typed register operand."
  def reg(type, id), do: {:reg, type, id}

  @doc "A predicate register operand."
  def pred(id), do: {:pred, id}

  @doc "An integer immediate."
  def imm(value), do: {:imm, value}

  @doc "A float immediate, emitted as a PTX hex float."
  def immf(type, value), do: {:immf, type, value}

  @doc "A kernel parameter operand."
  def param(name), do: {:param, name}

  @doc "A special register operand, e.g. `sreg(:ctaid_x)`."
  def sreg(name) when is_atom(name), do: {:sreg, name}

  @doc "A branch target."
  def label(name), do: {:label, name}

  @doc """
  Build an instruction.

      instr("ld", dtype: :u64, space: :param, dest: reg(:u64, 1), ops: [param(:v_out)])
      instr("ld", dtype: :f32, space: :global, vec: 4, dest: reg(:f32, 1),
            ops: [addr(reg(:u64, 2))])

  `vec` widens a load or store to `.v2`/`.v4`, which is how gpark gets vectorised
  memory traffic without asking a compiler to do it.
  """
  def instr(base, opts \\ []) do
    %{
      base: base,
      space: Keyword.get(opts, :space),
      modifier: Keyword.get(opts, :modifier),
      vec: Keyword.get(opts, :vec),
      dtype: Keyword.get(opts, :dtype),
      dest: Keyword.get(opts, :dest),
      ops: Keyword.get(opts, :ops, []),
      pred: Keyword.get(opts, :pred)
    }
  end

  @doc """
  A memory address operand, rendered `[base]`, `[base+offset]` or `[base+idx*scale]`.

      addr(reg(:u64, 2))                        # [%rd2]
      addr(reg(:u64, 2), 64)                    # [%rd2+64]
      addr(reg(:u64, 2), reg(:u64, 4))          # [%rd2+%rd4]
      addr(reg(:u64, 2), reg(:u64, 4), 4)       # [%rd2+%rd4*4]
      addr(param(:weights))                     # [weights]

  `idx` and `scale` are both optional, which keeps plain and strided addressing
  on one code path instead of two.
  """
  def addr(base, offset_or_idx \\ 0, scale \\ nil)
  def addr(base, offset, scale), do: {:addr, base, offset, scale}

  @doc "Build a basic block terminated by `term`."
  def block(label, instrs, term), do: %{label: label, instrs: instrs, term: term}

  @doc "Build a kernel."
  def kernel(name, opts \\ []) do
    %{
      name: name,
      target: Keyword.get(opts, :target, default_target()),
      ptx_version: Keyword.get(opts, :ptx_version, default_ptx_version()),
      params: Keyword.get(opts, :params, []),
      shared: Keyword.get(opts, :shared, 0),
      maxntid: Keyword.get(opts, :maxntid),
      blocks: Keyword.get(opts, :blocks, [])
    }
  end

  @doc "Build a kernel parameter descriptor."
  def param_decl(name, type, space \\ :global),
    do: %{name: name, type: type, space: space}

  @doc """
  Every register an instruction touches, as `{type, class, id}`.

  Includes operands, the destination, and any guarding predicate. The emitter
  uses this to size `.reg` declarations; the validator uses it to catch reads of
  registers that were never written.
  """
  def instr_regs(%{ops: ops, dest: dest, pred: pred}) do
    Enum.flat_map(ops, &operand_regs/1) ++ dest_regs(dest) ++ operand_regs(pred)
  end

  defp dest_regs(nil), do: []
  defp dest_regs({:reg, type, id}), do: [{type, reg_class(type), id}]
  defp dest_regs({:pred, id}), do: [{:pred, :p, id}]

  defp operand_regs({:reg, type, id}), do: [{type, reg_class(type), id}]
  defp operand_regs({:pred, id}), do: [{:pred, :p, id}]
  defp operand_regs(_), do: []

  @doc "Highest register id used per class, as a `%{class => id}` map."
  def max_regs(instrs) do
    instrs
    |> Enum.flat_map(&instr_regs/1)
    |> Enum.reduce(%{}, fn {_type, class, id}, acc ->
      Map.update(acc, class, id, &max(&1, id))
    end)
  end

  @doc "Highest predicate id used, or 0 when the kernel has no predicates."
  def max_preds(instrs) do
    instrs
    |> Enum.flat_map(&instr_regs/1)
    |> Enum.filter(&match?({:pred, _, _}, &1))
    |> Enum.reduce(0, fn {_, _, id}, acc -> max(acc, id) end)
  end
end