defmodule RiscGP.Emit do
  @moduledoc """
  RVGPU text emitter.

  Emits a stable, readable assembly-like text form. The output is *not* a
  machine encoding — the encoding is deliberately left open (see
  `isa/RVGPU-1.0-draft.md` §4) — so this is a readable form that round-trips
  through the corpus and can be diffed between the Elixir and Python
  implementations byte for byte.

  Every kernel is stamped with the ISA version and, when the ISA is not frozen,
  an `UNFROZEN` marker. That is a correctness feature, not decoration: a kernel
  built against a draft ISA that later gets retimed must not be mistaken for
  one built against a ratified encoding.

  ## Thread sections

  Instructions are grouped by issuing thread and emitted as
  `.thread <name>` sections, because on Path B the thread an instruction is
  issued from is semantically load-bearing — it is what lets a reader see the
  unpack/math/pack pattern directly in the listing.
  """

  alias RiscGP.IR
  alias RiscGP.Table

  @tab "  "

  @doc "Render a kernel to text."
  def emit(%{} = kernel), do: kernel |> sections() |> IO.iodata_to_binary()

  @doc "Render and write to `path`."
  def emit!(%{} = kernel, path) do
    text = emit(kernel)
    File.write!(path, text)
    text
  end

  defp sections(kernel) do
    [
      header(kernel),
      params(kernel),
      body(kernel)
    ]
  end

  # ---------------------------------------------------------------------------

  defp header(kernel) do
    {isa, version} = Table.isa()

    [
      "; riscgp RVGPU kernel\n",
      "; isa: ", isa, " ", version, "\n",
      frozen_marker(),
      "; path: ", path_text(kernel.path), "\n",
      "\n"
    ]
  end

  defp frozen_marker do
    if Table.frozen?() do
      ""
    else
      ["; WARNING: ISA ", elem(Table.isa(), 1), " is ", Table.status(), "\n",
       "; timings may change; do not ship binaries built from this\n"]
    end
  end

  defp path_text(:a), do: "A (RVV datapath)"
  defp path_text(:b), do: "B (RV cores + coprocessor)"

  defp params(%{params: []}), do: []

  defp params(%{params: params}) do
    entries = Enum.map(params, fn p -> [@tab, "param ", to_string(p.name), " : ", to_string(p.type)] end)
    [".params\n", Enum.intersperse(entries, "\n"), "\n\n"]
  end

  defp body(kernel) do
    grouped = Enum.group_by(kernel.blocks, & &1.thread)

    Enum.flat_map(IR.threads(), fn thread ->
      case Map.get(grouped, thread) do
        nil -> []
        blocks -> [".thread ", to_string(thread), "\n", Enum.flat_map(blocks, &block/1), "\n"]
      end
    end)
  end

  defp block(%{label: label, instrs: instrs, term: term}) do
    [
      label_text(label), ":\n",
      Enum.map(instrs, fn i -> [@tab, line(i)] end),
      if(term, do: [@tab, line(term)], else: [])
    ]
  end

  defp line(instr) do
    [opcode(instr), args(instr), "    ; ", provenance(instr), "\n"]
  end

  # Provenance on every line. This is the whole point of the ISA's `domain`
  # tag: a reader can see at a glance that a `mop.mma` does not run where a
  # `dm.push` runs, which is the fact that makes the async hazard in §7 of the
  # spec visible instead of folklore.
  defp provenance(instr) do
    "#{instr.thread}/#{Table.unit(instr.base)} lat=#{Table.latency(instr.base)} thr=#{Table.throughput(instr.base)}"
  end

  # ---------------------------------------------------------------------------
  # Opcode
  # ---------------------------------------------------------------------------

  @doc """
  Render an instruction's opcode, dotted parts in table order.

      iex> RiscGP.Emit.opcode(%{base: "mop.mma", modifier: "acc", space: nil, dtype: :f32})
      "mop.mma.acc.f32"

      iex> RiscGP.Emit.opcode(%{base: "lw", modifier: nil, space: :sram, dtype: :f32})
      "lw.sram.f32"
  """
  def opcode(%{base: base} = instr) do
    spec = Table.op_spec(base) || raise ArgumentError, "unknown opcode #{inspect(base)}"

    Enum.map_join(spec.parts, fn
      :name -> base
      :space -> if instr.space, do: "." <> to_string(instr.space), else: ""
      :modifier -> if instr.modifier, do: "." <> instr.modifier, else: ""
      :dtype -> if instr.dtype, do: "." <> to_string(instr.dtype), else: ""
    end)
  end

  defp args(%{dest: nil, ops: []}), do: " "

  defp args(%{dest: dest, ops: []}) do
    [" ", operand(dest), " "]
  end

  defp args(%{dest: nil, ops: ops}) do
    [" ", Enum.map_join(ops, ", ", &operand/1), " "]
  end

  defp args(%{dest: dest, ops: ops}) do
    [" ", operand(dest), ", ", Enum.map_join(ops, ", ", &operand/1), " "]
  end

  # ---------------------------------------------------------------------------
  # Operands
  # ---------------------------------------------------------------------------

  @doc """
  Render one operand.

      iex> RiscGP.Emit.operand(RiscGP.IR.reg(:u32, 5))
      "x5"

      iex> RiscGP.Emit.operand(RiscGP.IR.dst(0))
      "dst0"

      iex> RiscGP.Emit.operand(RiscGP.IR.addr(RiscGP.IR.reg(:u32, 5), RiscGP.IR.reg(:u32, 6), 64))
      "[x5+x6*64]"
  """
  def operand({:reg, _type, id}), do: "x" <> Integer.to_string(id)
  def operand({tag, id}) when tag in [:dst, :lreg, :srca, :srcb, :vec, :sem, :cb], do: "#{tag}#{id}"
  def operand({:imm, value}) when is_integer(value), do: Integer.to_string(value)
  def operand({:param, name}), do: "[" <> to_string(name) <> "]"
  def operand({:label, name}), do: label_text(name)

  def operand({:addr, base, nil, nil}), do: ["[", operand(base), "]"]

  def operand({:addr, base, idx, nil}) when is_integer(idx) do
    ["[", operand(base), "+", Integer.to_string(idx), "]"]
  end

  def operand({:addr, base, idx, nil}), do: ["[", operand(base), "+", operand(idx), "]"]

  def operand({:addr, base, idx, scale}) do
    ["[", operand(base), "+", operand(idx), "*", Integer.to_string(scale), "]"]
  end

  defp label_text(label), do: ".L_" <> to_string(label)
end
