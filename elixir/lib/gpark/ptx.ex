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

  @behaviour Gpark.Backend

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
  @impl Gpark.Backend
  def emit(%{} = kernel), do: kernel |> sections() |> IO.iodata_to_binary()

  @doc "A short name for this backend."
  @impl Gpark.Backend
  def name, do: :ptx

  @doc """
  Every opcode base gpark can render.

  The table is the single source of truth, so this cannot drift from what `emit/1`
  is willing to spell. A backend whose list was hand-kept would eventually claim an
  opcode it refuses and reject one it can handle, and the gate would be worse than
  no gate.
  """
  @impl Gpark.Backend
  def ops, do: Gpark.Ops.names()

  @doc """
  Every type gpark can represent.

  Includes the four sub-byte types, which have no direct PTX spelling: `ptx_type/1`
  returns `nil` for a bare `:u4` because PTX cannot name one. They are declared
  supported because they are representable rather than merely named -- packed into a
  container (`Type.packed(:u32, :u4, 8)`, whose `ptx_type/1` is `:u32`) or widened
  before arithmetic (`Type.widen(:u4) == :u16`). Dropping them from this list would
  make `Backend.require!/2` reject every sub-byte kernel on a technicality.
  """
  @impl Gpark.Backend
  def types, do: Gpark.Type.all()

  @doc """
  Structural validation, delegated to `Gpark.Validate`.

  Separate from `Gpark.Backend.require!/2` on purpose: this reports a kernel that is
  malformed, where `require!/2` reports a kernel that is fine but outside this
  backend's capabilities. A kernel can pass one and fail the other.
  """
  @impl Gpark.Backend
  def check(kernel), do: Gpark.Validate.check(kernel)

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
  #
  # One declaration per register, not the `%rd1<4>` vector form. A real ptxas
  # (12.8) rejects the vector form outright, and the rejection is not obvious
  # from the error: it reports "Arguments mismatch" on the first *instruction*,
  # not on the declaration, so it cascades through the whole kernel and looks
  # like an addressing bug. Confirmed with remote/ptx_probe.sh, where every
  # `.reg .T %base<N>;` case fails and every single-register case passes.
  # One per line is also the honest encoding: this design allocates explicitly
  # and never reuses, so there is nothing to gain by declaring a vector.
  defp declarations(_kernel, []) do
    []
  end

  defp declarations(_kernel, instrs) do
    groups =
      instrs
      |> Enum.flat_map(&IR.instr_regs/1)
      |> Enum.group_by(fn {type, class, _id} -> {class, type} end)
      |> Enum.map(fn {{class, type}, regs} ->
        ids = regs |> Enum.map(&elem(&1, 2)) |> Enum.uniq() |> Enum.sort()
        {@class_order[class], class, type, ids}
      end)
      |> Enum.sort()

    Enum.flat_map(groups, fn {_order, class, type, ids} ->
      Enum.map(ids, fn id ->
        [@tab, ".reg .", to_string(type), " ", register(class, id), ";\n"]
      end)
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

  # A register offset is not expressible as a PTX addressing mode and used to
  # be rendered inline as `[base+idx]`. PTX does not accept it: `[%rd+%rd]`,
  # `[%rd+%r]` and `[%r+%rd]` all fail to parse in ptxas 12.8, because ld/st
  # take `[reg]` or `[reg+imm]` only. An index that is not known until launch
  # cannot become an immediate either, so the only correct encoding is to build
  # the address in a register first:
  #
  #     add.s64 %rd_addr, %rd_base, %rd_off;
  #     ld.global.f32 %f, [%rd_addr];
  #
  # That is deliberately not hidden behind an extra pass in this module. This
  # design has no hidden allocation, so the register is allocated explicitly by
  # whoever builds the IR and the address arithmetic is visible in the IR. See
  # Gpark.DSL.address_materialised/3.
  def operand({:addr, base, idx, nil}) do
    raise ArgumentError, """
    register offsets cannot be rendered as a PTX addressing mode: #{operand(base)}+#{operand(idx)}.

    PTX ld/st accept [reg] or [reg+imm] only; register+register does not parse.
    Materialise the address into an explicitly allocated register instead:

        add.s64 <addr_reg>, #{operand(base)}, #{operand(idx)};
        ld.global.f32 <dst>, [<addr_reg>];
    """
  end

  def operand({:addr, base, idx, scale}) do
    raise ArgumentError, """
    strided addresses (#{operand(base)}+#{operand(idx)}*#{scale}) cannot be
    rendered as a PTX addressing mode. PTX ld/st accept [reg] or [reg+imm] only.

    Widen the index, then materialise the address into an explicitly allocated
    register:

        mul.wide.s32 <off_reg>, #{operand(idx)}, #{scale};
        add.s64 <addr_reg>, #{operand(base)}, <off_reg>;
        ld.global.f32 <dst>, [<addr_reg>];
    """
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

  defp class_prefix(:p), do: "p"
  defp class_prefix(:r), do: "r"
  defp class_prefix(:rd), do: "rd"
  defp class_prefix(:f), do: "f"
  defp class_prefix(:fd), do: "fd"

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
