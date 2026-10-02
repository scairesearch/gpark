defmodule Gpark.PTX do
  @moduledoc """
  PTX emitter for `Gpark.IR`.

  Renders a kernel to PTX text. The output style deliberately imitates `nvcc`
  output — tab indentation, contiguous register ranges like `%f<8>`, hex float
  immediates — so that `ptxas` accepts it and so goldens in `corpus/` can be
  diffed against what the CUDA toolchain produces for an equivalent kernel.

  The emitter is *unopinionated about correctness*. It will happily emit a
  kernel that `Gpark.Validate` rejects, because the point of gpark is to get
  close to the metal. Run validation before you run `ptxas`; see
  `docs/VALIDATION.md`.
  """

  alias Gpark.IR

  # PTX has no fixed register classes, but nvcc's ordering is a good convention
  # to follow: predicates, then 32-bit, then 64-bit, then the float banks.
  @class_order %{p: 0, r: 1, rd: 2, f: 3, fd: 4}

  @tab "\t"

  @doc """
  Render a kernel to PTX text.

      iex> k = Gpark.IR.kernel(:noop, blocks: [...])
      iex> Gpark.PTX.emit(k) =~ ".target sm_80"
      true
  """
  def emit(%{} = kernel), do: kernel |> sections() |> IO.iodata_to_binary()

  @doc "Render a kernel and write it to `path`."
  def emit!(%{} = kernel, path) do
    ptx = emit(kernel)
    File.write!(path, ptx)
    ptx
  end

  defp sections(kernel) do
    [
      header(kernel),
      signature(kernel),
      body(kernel)
    ]
  end

  # ---------------------------------------------------------------------------
  # Module header
  # ---------------------------------------------------------------------------

  defp header(kernel) do
    [
      ".version ",
      to_string(kernel.ptx_version),
      "\n",
      ".target ",
      to_string(kernel.target),
      "\n",
      ".address_size 64\n"
    ]
  end

  defp signature(%{name: name, params: []}) do
    [".visible .entry ", to_string(name), "()\n"]
  end

  defp signature(%{name: name, params: params}) do
    [
      ".visible .entry ",
      to_string(name),
      "(\n",
      params
      |> Enum.map(fn p -> [@tab, ".param .", to_string(p.type), " ", to_string(p.name)] end)
      |> Enum.intersperse(",\n"),
      "\n)\n"
    ]
  end

  # ---------------------------------------------------------------------------
  # Function body
  # ---------------------------------------------------------------------------

  defp body(kernel) do
    instrs = all_instrs(kernel)

    [
      "{\n",
      declarations(kernel, instrs),
      shared_decl(kernel),
      Enum.map(kernel.blocks, &block/1),
      "}\n"
    ]
  end

  # `.reg` declarations sized from actual usage, so the kernel never allocates a
  # register it does not need — on a register-starved quant kernel that spare
  # vector is a whole extra load in flight.
  defp declarations(_kernel, []) do
    []
  end

  defp declarations(_kernel, instrs) do
    groups =
      instrs
      |> Enum.flat_map(&IR.instr_regs/1)
      |> Enum.group_by(fn {type, class, _id} -> {class, type} end)
      |> Enum.map(fn {{class, type}, regs} ->
        ids = regs |> Enum.map(&elem(&1, 2)) |> Enum.uniq()
        {@class_order[class], class, type, ids}
      end)
      |> Enum.sort()

    Enum.map(groups, fn {_order, class, type, ids} ->
      [@tab, ".reg .", to_string(type), " ", register(class, ids), ";\n"]
    end)
  end

  # Shared scratch is declared in the body, which is where PTX requires it.
  defp shared_decl(%{shared: 0}), do: []

  defp shared_decl(%{shared: n}) do
    [@tab, ".extern .shared .align 4 .b8 __gpark_shared[", Integer.to_string(n), "];\n"]
  end

  defp block(%{label: label, instrs: instrs, term: term}) do
    [
      label_prefix(label),
      ":\n",
      Enum.map(instrs, fn i -> [@tab, line(i)] end),
      if(term, do: [@tab, line(term)], else: [])
    ]
  end

  defp line(%{
         base: base,
         space: space,
         modifier: modifier,
         vec: vec,
         dtype: dtype,
         srctype: srctype,
         dest: dest,
         ops: ops,
         pred: pred
       }) do
    opcode =
      opcode(%{
        base: base,
        space: space,
        modifier: modifier,
        vec: vec,
        dtype: dtype,
        srctype: srctype
      })

    [guard(pred), opcode, join_args(dest, ops), ";\n"]
  end

  # PTX puts exactly one space after the opcode and separates the optional
  # destination from the operand list with ", " — matching nvcc so goldens stay
  # diffable against real CUDA output rather than against our own formatting.
  defp join_args(nil, []), do: ""
  defp join_args(nil, ops), do: [" ", operand_list(ops)]
  defp join_args(dest, []), do: [" ", operand(dest)]
  defp join_args(dest, ops), do: [" ", operand(dest), ", ", operand_list(ops)]

  defp operand_list(ops), do: Enum.map_join(ops, ", ", &operand/1)

  # `@%p` prefixes a predicated statement; PTX has no "branch if false", so a
  # negated predicate is expressed by negating it explicitly upstream.
  defp guard(nil), do: ""
  defp guard(pred), do: ["@", operand(pred), " "]

  # ---------------------------------------------------------------------------
  # Opcodes
  # ---------------------------------------------------------------------------

  @doc """
  Render an instruction's opcode string, dotted parts in PTX order.

      iex> Gpark.PTX.opcode(%{base: "ld", space: :global, modifier: nil, vec: nil, dtype: :f32})
      "ld.global.f32"
  """
  def opcode(%{base: base} = instr) do
    spec = IR.op_spec(base) || raise ArgumentError, "unknown opcode #{inspect(base)}"

    Enum.map_join(spec.parts, fn
      :name -> base
      :space -> if instr.space, do: [".", to_string(instr.space)], else: ""
      :sync -> ".sync"
      :modifier -> if instr.modifier, do: [".", instr.modifier], else: ""
      :vec -> if instr.vec, do: [".v", Integer.to_string(instr.vec)], else: ""
      :dtype -> if instr.dtype, do: [".", to_string(instr.dtype)], else: ""
      :srctype -> if instr.srctype, do: [".", to_string(instr.srctype)], else: ""
    end)
  end

  # ---------------------------------------------------------------------------
  # Operands
  # ---------------------------------------------------------------------------

  @doc """
  Render one operand to a string.

  Every operand clause returns a binary, not iodata. `Enum.map_join` needs a
  binary here, and making the public function return one keeps it assertable.

      iex> Gpark.PTX.operand({:reg, :f32, 1})
      "%f1"
  """
  def operand({:reg, type, id}), do: register(IR.reg_class(type), id)
  def operand({:pred, id}), do: "%p#{id}"
  def operand({:imm, value}) when is_integer(value), do: Integer.to_string(value)
  def operand({:immf, type, value}), do: hex_float(value, type)
  def operand({:param, name}), do: "[" <> to_string(name) <> "]"
  def operand({:sreg, name}), do: "%" <> to_string(Map.fetch!(IR.sreg_names(), name))
  def operand({:label, name}), do: label_prefix(name)
  # base only
  def operand({:addr, base, {:imm, 0}, nil}), do: "[" <> operand(base) <> "]"

  # base + constant byte offset
  def operand({:addr, base, {:imm, offset}, nil}) when is_integer(offset) do
    "[" <> operand(base) <> "+" <> Integer.to_string(offset) <> "]"
  end

  # base + register (index held in a register)
  def operand({:addr, base, idx, nil}), do: "[" <> operand(base) <> "+" <> operand(idx) <> "]"

  # strided: base + idx * scale, the shape every tiled kernel wants
  def operand({:addr, base, idx, scale}) do
    "[" <> operand(base) <> "+" <> operand(idx) <> "*" <> Integer.to_string(scale) <> "]"
  end

  @doc """
  Render a PTX hex float immediate.

  PTX rejects bare floating-point literals in most contexts, so f32 must be
  written as `0f` plus 8 hex digits and f64 as `0d` plus 16.

      iex> Gpark.PTX.hex_float(1.0, :f32)
      "0f3F800000"
  """
  def hex_float(value, :f32) do
    <<bits::unsigned-big-size(32)>> = <<value::float-big-size(32)>>
    "0f" <> hex(bits, 8)
  end

  def hex_float(value, :f64) do
    <<bits::unsigned-big-size(64)>> = <<value::float-big-size(64)>>
    "0d" <> hex(bits, 16)
  end

  def hex_float(_value, type) do
    raise ArgumentError, "no hex float encoding for #{inspect(type)}"
  end

  # ---------------------------------------------------------------------------
  # Registers and labels
  # ---------------------------------------------------------------------------

  defp register(nil, _id), do: raise(ArgumentError, "operand type has no register class")

  # One register: `%f3`.
  defp register(class, id) when is_integer(id) do
    "%" <> class_prefix(class) <> Integer.to_string(id)
  end

  # A `.reg` declaration listing: `%f1<3>`.
  defp register(class, ids) when is_list(ids) do
    prefix = "%" <> class_prefix(class)

    ids
    |> compress()
    |> Enum.map_join(", ", fn {first, last} -> prefix <> run_text(first, last) end)
  end

  defp run_text(id, id), do: Integer.to_string(id)
  defp run_text(first, last), do: "#{first}<#{last - first + 1}>"

  defp class_prefix(:p), do: "p"
  defp class_prefix(:r), do: "r"
  defp class_prefix(:rd), do: "rd"
  defp class_prefix(:f), do: "f"
  defp class_prefix(:fd), do: "fd"

  # Compress a sorted id list into PTX's `[%r<4><2>]`-style runs. Used so
  # declarations look like nvcc output rather than a wall of single registers.
  defp compress(ids) do
    {runs, current} =
      ids
      |> Enum.sort()
      |> Enum.reduce({[], nil}, fn id, {acc, cur} ->
        case cur do
          nil -> {acc, {id, id}}
          {first, last} when id <= last + 1 -> {acc, {first, id}}
          {first, last} -> {[{first, last} | acc], {id, id}}
        end
      end)

    case current do
      nil -> Enum.reverse(runs)
      {first, last} -> Enum.reverse([{first, last} | runs])
    end
  end

  defp hex(value, width) do
    value |> Integer.to_string(16) |> String.upcase() |> String.pad_leading(width, "0")
  end

  # `$`-prefixed labels match nvcc's convention and keep block labels from
  # colliding with kernel parameter names.
  defp label_prefix(label), do: "$L__" <> to_string(label)

  # nvcc indents instruction bodies with a tab; matching it keeps goldens
  # diffable against real CUDA output instead of against our own formatting.

  defp all_instrs(%{blocks: blocks}) do
    Enum.flat_map(blocks, fn b -> b.instrs ++ List.wrap(b.term) end)
  end
end
