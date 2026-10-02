defmodule RiscGP.IR do
  @moduledoc """
  Kernel IR for RVGPU.

  Deliberately explicit about machine state, in the same spirit as gpark: on a
  machine this asynchronous, an IR that quietly reorders or hides the thread
  and unit each instruction belongs to is working against the programmer. Every
  instruction carries a `thread`, and the unit it runs on is looked up from
  `RiscGP.Table` rather than declared.

  Everything is a plain map so kernels round-trip through JSON, which is what
  lets `corpus/` hold portable specs both implementations read.

  ## Shapes

      kernel  %{name: atom, path: :a | :b, params: [param], blocks: [block]}

      block   %{label: atom, thread: thread, instrs: [instr], term: instr | nil}

      instr   %{base: string, thread: thread, modifier: string | nil,
                space: atom | nil, dtype: atom | nil,
                dest: operand | nil, ops: [operand]}

  ## Operands

      {:reg, type, id}    core GPR
      {:dst, id}          matrix accumulator
      {:lreg, id}         left matrix operand staging
      {:srca, id}         right operand A buffer
      {:srcb, id}         right operand B buffer
      {:vec, id}          RVV vector register (Path A)
      {:sem, id}          tile semaphore
      {:cb, id}           circular-buffer descriptor
      {:imm, value}       integer immediate
      {:param, name}      kernel parameter
      {:addr, base, idx, scale}  base + idx * scale, all four slots always present
      {:label, atom}      branch target

  ## Threads

      :b    data movement / control (RV-B in Tenstorrent naming)
      :t0   unpack thread   — feeds the Matrix Unit
      :t1   math thread     — drives the Matrix Unit
      :t2   pack thread     — drains the Matrix Unit
      :nc   no-network core — data movement, poor coprocessor access
  """

  @type thread :: :b | :t0 | :t1 | :t2 | :nc

  @threads [:b, :t0, :t1, :t2, :nc]

  @doc "The five threads of a Tensix tile."
  def threads, do: @threads

  @reg_files ~w(reg dst lreg srca srcb vec sem cb)

  @doc "Register files an operand can name."
  def reg_files, do: @reg_files

  @doc "Address spaces the ISA defines."
  def address_spaces, do: RiscGP.Table.address_spaces()

  @doc "Data types the ISA defines, derived from the opcode table."
  def dtypes do
    RiscGP.Table.opcodes()
    |> Enum.flat_map(fn {_n, spec} -> spec.dtypes end)
    |> Enum.uniq()
    |> Enum.sort()
  end

  # ---------------------------------------------------------------------------
  # Constructors
  # ---------------------------------------------------------------------------

  def thread(name) when name in @threads, do: name

  @doc "A core GPR operand."
  def reg(type, id), do: {:reg, type, id}

  @doc "A matrix accumulator."
  def dst(id), do: {:dst, id}

  @doc "Left matrix operand staging register."
  def lreg(id), do: {:lreg, id}

  @doc "Right operand A buffer."
  def srca(id), do: {:srca, id}

  @doc "Right operand B buffer."
  def srcb(id), do: {:srcb, id}

  @doc "An RVV vector register. Path A only."
  def vec(id), do: {:vec, id}

  @doc "A tile semaphore."
  def sem(id), do: {:sem, id}

  @doc "A circular-buffer descriptor."
  def cb(id), do: {:cb, id}

  @doc "An integer immediate."
  def imm(value), do: {:imm, value}

  @doc "A kernel parameter."
  def param(name), do: {:param, name}

  @doc "A branch target."
  def label(name), do: {:label, name}

  @doc """
  A memory address, `base + idx * scale`.

      addr(reg(:u32, 5))            # [x5]
      addr(reg(:u32, 5), 4)         # [x5+4]
      addr(reg(:u32, 5), reg(:u32, 6))  # [x5+x6]
      addr(reg(:u32, 5), reg(:u32, 6), 64) # [x5+x6*64]

  All four slots are always present — `scale` is `nil` rather than `1` when
  absent, so a missing scale is visible in the IR rather than silently meaning
  one thing in the emitter and another in the cycle model.
  """
  def addr(base, idx \\ nil, scale \\ nil), do: {:addr, base, idx, scale}

  @doc """
  Build an instruction.

      instr("mop.mma", thread: :t1, modifier: "acc", dtype: :f32,
            dest: dst(0), ops: [lreg(0), srca(0), srcb(0)])
  """
  def instr(base, opts \\ []) do
    %{
      base: base,
      thread: Keyword.get(opts, :thread, :t1),
      modifier: Keyword.get(opts, :modifier),
      space: Keyword.get(opts, :space),
      dtype: Keyword.get(opts, :dtype),
      dest: Keyword.get(opts, :dest),
      ops: Keyword.get(opts, :ops, [])
    }
  end

  @doc "Build a basic block owned by `thread`."
  def block(label, thread, instrs, term), do: %{label: label, thread: thread, instrs: instrs, term: term}

  @doc "Build a kernel. `path` is `:a` or `:b`."
  def kernel(name, opts \\ []) do
    %{
      name: name,
      path: Keyword.get(opts, :path, :b),
      params: Keyword.get(opts, :params, []),
      blocks: Keyword.get(opts, :blocks, [])
    }
  end

  @doc "Build a kernel parameter descriptor."
  def param_decl(name, type), do: %{name: name, type: type}

  # ---------------------------------------------------------------------------
  # Introspection
  # ---------------------------------------------------------------------------

  @doc "Every register an instruction touches, as `{file, id}`."
  def instr_regs(%{dest: dest, ops: ops}) do
    Enum.flat_map(ops, &operand_regs/1) ++ dest_regs(dest)
  end

  defp dest_regs(nil), do: []
  defp dest_regs({tag, id}) when tag in @reg_files, do: [{tag, id}]

  defp operand_regs({tag, id}) when tag in @reg_files, do: [{tag, id}]
  defp operand_regs({:addr, base, idx, _scale}), do: operand_regs(base) ++ operand_regs(idx)
  defp operand_regs(_), do: []

  @doc "The register file a tagged operand belongs to, or nil."
  def reg_file({tag, _id}) when tag in @reg_files, do: tag
  def reg_file({:reg, _type, _id}), do: :reg
  def reg_file(_), do: nil

  @doc "Every instruction in a kernel, blocks and terminators alike."
  def all_instrs(%{blocks: blocks}) do
    Enum.flat_map(blocks, fn b -> b.instrs ++ List.wrap(b.term) end)
  end
end
