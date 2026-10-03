defmodule Gpark.DSL.Scope do
  @moduledoc """
  The state threaded through a `Gpark.DSL.elementwise/2` body.

  This is the whole ergonomic claim of the DSL, and it is worth being precise about
  what it does and does not do.

  It allocates registers, monotonically, per PTX register class. It does *not*
  coalesce, reorder, spill, or reuse. That is deliberate: gpark's premise is that
  register pressure and scheduling are the levers on a memory-bound kernel, so an
  allocator that quietly recycles a register across a `.bar.sync` would destroy the
  thing the user is trying to control.

  Allocation order is first-definition order, which is also what a careful human
  writes by hand. `Gpark.DSLTest` pins that: the DSL reproduces the hand-written
  `vec_add_f32` register plan exactly.
  """

  @type t :: %__MODULE__{}

  defstruct instrs: [],
            counter: %{},
            ptrs: %{},
            index: nil,
            index_type: :u32,
            offset: nil,
            addr_reg: nil
end

defmodule Gpark.DSL do
  @moduledoc """
  A front end for the shape of kernel gpark keeps writing by hand.

  ## What this is for

  In `Gpark.Kernels.VecAddF32`, most of the code is not the arithmetic. It is:
  loading four parameters out of the driver bank, computing a thread index,
  establishing a bounds guard, turning an element index into a byte offset, and
  emitting a second block so the guard has somewhere to branch to. That boilerplate
  is mechanical to write and easy to get subtly wrong — and its failure mode is an
  out-of-bounds write, which is the bug `unpack_u4_f32` actually shipped until it was
  caught.

  `elementwise/2` owns that boilerplate and hands the body a typed scope. The
  arithmetic stays explicit.

  ## What this deliberately does not do

  It introduces no implicit tensors, shapes, strides or dtype inference. Those are
  the features that make a DSL pleasant and also make it impossible to reason about
  the PTX underneath. Here a pointer is a `u64` plus an element type, `count` is a
  `u32`, and that is the entire type system.

  It does not bypass `Gpark.IR`. The result is an ordinary IR map, and it goes
  through `Gpark.Validate` and `Gpark.Backend.require!/2` like anything else.

  ## Usage

      import Gpark.DSL

      Gpark.DSL.elementwise(:vec_add_f32,
        pointers: [a: :f32, b: :f32, out: :f32],
        count: :n,
        index: :ctaid_x,
        body: fn k ->
          {k, a} = load(k, :a)
          {k, b} = load(k, :b)
          {k, sum} = op(k, :add, :f32, [a, b])
          store(k, :out, sum)
        end
      )

  One thread per element, with the element index taken straight from `ctaid.x`.

  The scope is threaded as the first argument rather than hidden in process state or
  a closure. A DSL that carried a mutable builder implicitly would be shorter to
  write, but then a kernel could not be built twice in one process, two bodies could
  not interleave, and a failure inside `body` would leave the builder somewhere
  unspecified. Threading it costs one extra argument and keeps the construction
  re-entrant.
  """

  alias Gpark.IR
  alias Gpark.DSL.Scope

  @doc """
  Build a bounds-checked elementwise kernel.

  Options:

    * `:pointers` — `[name: elem_type]`, or `{name, elem_type, space}`. Emitted as
      `u64` parameter declarations and loaded in the order given.
    * `:count` — name of the `u32` length parameter. The guard is `index >= count`,
      branching to `:guard_label`. That is the shape the corpus uses: an explicit
      early exit rather than a predicated load/store pair, so every register read
      still has an obvious write before it.
    * `:index` — special register supplying the element index. Default `:ctaid_x`.
    * `:index_type` — type of the index. Default `:u32`.
    * `:body` — `fn scope -> nil end`. Instructions are emitted in call order.
    * `:guard_label` — name of the epilogue block. Default `:done`.
    * `:target`, `:ptx_version` — passed through to `Gpark.IR.kernel/2`.

  Parameter order is the pointers in declaration order followed by `:count`. The
  corpus goldens depend on that, so it is part of the contract.
  """
  @spec elementwise(atom(), keyword()) :: map()
  def elementwise(name, opts) do
    pointers = normalize_pointers(Keyword.fetch!(opts, :pointers))
    count = Keyword.fetch!(opts, :count)
    index_sreg = Keyword.get(opts, :index, :ctaid_x)
    index_type = Keyword.get(opts, :index_type, :u32)
    guard_label = Keyword.get(opts, :guard_label, :done)
    body = Keyword.fetch!(opts, :body)

    scope = %Scope{index_type: index_type}

    params =
      Enum.map(pointers, fn {pname, _elem, _space} -> IR.param_decl(pname, :u64) end) ++
        [IR.param_decl(count, :u32)]

    # Load every pointer, then the count. `alloc/2` numbers them in this order, so
    # the resulting register plan is fixed by the declaration order above.
    scope =
      Enum.reduce(pointers, scope, fn {pname, elem, space}, acc ->
        {acc, base} =
          emit_dest(acc, "ld", :u64, space: :param, dtype: :u64, ops: [IR.param(pname)])

        # Remember which register holds this pointer's base address. The byte offset
        # is shared across pointers, but the base is per-pointer: conflating the two
        # emits `[offset]` alone, which reads from address 4 instead of `a + 4`.
        %{acc | ptrs: Map.put(acc.ptrs, pname, {elem, space, base})}
      end)

    {scope, count_reg} =
      emit_dest(scope, "ld", :u32, space: :param, dtype: :u32, ops: [IR.param(count)])

    {scope, index_reg} =
      emit_dest(scope, "mov", index_type, dtype: index_type, ops: [IR.sreg(index_sreg)])

    scope = %{scope | index: index_reg}

    {scope, p_in} =
      emit_dest(scope, "setp", :pred,
        modifier: "ge",
        dtype: :u32,
        ops: [IR.reg(index_type, index_reg), IR.reg(:u32, count_reg)]
      )

    {scope, p_done} = emit_dest(scope, "not", :pred, dtype: :pred, ops: [IR.pred(p_in)])

    scope =
      emit(scope, "bra", ops: [IR.label(guard_label)], pred: IR.pred(p_done))

    # One `mul.wide` for the whole kernel, reused by every access. Widening here is
    # the difference between a kernel that stays correct past 2^30 elements and one
    # that wraps; a plain `.u32` multiply truncates the byte offset.
    elem = first_elem_type(pointers)

    {scope, offset_reg} =
      emit_dest(scope, "mul.wide", :u64,
        dtype: :u32,
        ops: [IR.reg(index_type, index_reg), IR.imm(byte_width(elem))]
      )

    scope = %{scope | offset: offset_reg}
    scope = body.(scope)

    IR.kernel(name,
      target: Keyword.get(opts, :target, IR.default_target()),
      ptx_version: Keyword.get(opts, :ptx_version, IR.default_ptx_version()),
      params: params,
      blocks: [
        IR.block(:entry, scope.instrs, IR.instr("ret")),
        IR.block(guard_label, [], IR.instr("ret"))
      ]
    )
  end

  # ---------------------------------------------------------------------------
  # Scope API, called from inside the body
  # ---------------------------------------------------------------------------

  @doc "The element index register, as an operand of the index type."
  @spec idx(Scope.t()) :: tuple()
  def idx(%Scope{index: index, index_type: type}), do: IR.reg(type, index)

  @doc "The shared byte-offset register, as a `u64` operand."
  @spec offset(Scope.t()) :: tuple()
  def offset(%Scope{offset: offset}), do: IR.reg(:u64, offset)

  @doc """
  Load one element from pointer `:name`.

  The destination takes the pointer's declared element type, so a load's dtype and
  the register it lands in cannot disagree.
  """
  @spec load(Scope.t(), atom()) :: {Scope.t(), tuple()}
  def load(%Scope{} = scope, name) do
    {elem, space, _base} = fetch_ptr!(scope, name)
    {scope, addr} = addr_of(scope, name)

    {scope, id} =
      emit_dest(scope, "ld", elem, dtype: elem, space: space, ops: [addr])

    {scope, IR.reg(elem, id)}
  end

  @doc "Store `value` to one element of pointer `:name`."
  @spec store(Scope.t(), atom(), tuple()) :: Scope.t()
  def store(%Scope{} = scope, name, value) do
    {_elem, space, _base} = fetch_ptr!(scope, name)
    {scope, addr} = addr_of(scope, name)

    emit(scope, "st",
      dtype: operand_type(value),
      space: space,
      ops: [addr, value]
    )
  end

  @doc """
  Emit an instruction with an automatically allocated destination.

  `ops` must already be IR operand tuples. Use this for anything the sugar does not
  cover — `cvt`, shifts, the bit twiddling an unpack needs — so the DSL does not
  grow a second, narrower opcode vocabulary than `Gpark.IR` already has.
  """
  @spec op(Scope.t(), binary(), atom(), [tuple()]) :: {Scope.t(), tuple()}
  def op(%Scope{} = scope, base, dtype, ops) do
    {scope, id} = emit_dest(scope, base, dtype, dtype: dtype, ops: ops)
    {scope, IR.reg(dtype, id)}
  end

  @doc "Allocate a register of `type` without emitting anything."
  @spec temp(Scope.t(), atom()) :: {Scope.t(), tuple()}
  def temp(%Scope{} = scope, type) do
    {scope, id} = alloc(scope, type)
    {scope, IR.reg(type, id)}
  end

  @doc "A kernel-parameter operand."
  @spec param(atom()) :: tuple()
  def param(name), do: IR.param(name)

  @doc "An integer immediate operand."
  @spec imm(integer()) :: tuple()
  def imm(value), do: IR.imm(value)

  # ---------------------------------------------------------------------------
  # Internals
  # ---------------------------------------------------------------------------

  defp normalize_pointers(list) when is_list(list) do
    Enum.map(list, fn
      {name, elem} -> {name, elem, :global}
      {name, elem, space} -> {name, elem, space}
    end)
  end

  defp first_elem_type([{_name, elem, _space} | _]), do: elem

  defp byte_width(elem), do: div(IR.width(elem), 8)

  defp fetch_ptr!(%Scope{ptrs: ptrs}, name) do
    case Map.fetch(ptrs, name) do
      {:ok, pair} ->
        pair

      :error ->
        raise ArgumentError,
              "no pointer #{inspect(name)} declared; have: #{inspect(Map.keys(ptrs))}"
    end
  end

  # Every pointer shares one byte-offset register, which is the point: for an
  # elementwise kernel the offsets are identical, so recomputing them per access
  # would be instructions for nothing.
  #
  # The address itself is then materialised into one dedicated register, also
  # shared across accesses. PTX has no base+register addressing mode -- ld/st
  # take `[reg]` or `[reg+imm]` and nothing else -- so the sum has to exist in a
  # register before the load can name it. Reusing a single register across
  # accesses rather than allocating one per access costs one u64 per kernel
  # instead of one per ld/st, and the register count it produces is reported by
  # `remote/ptxas_check.sh`, so the saving is measured rather than assumed.
  #
  # Allocated here rather than in the backend so it stays visible in the IR:
  # this design does not hide allocation.
  defp addr_of(%Scope{} = scope, name) do
    {_elem, _space, base} = fetch_ptr!(scope, name)
    {scope, id} = materialize_addr(scope)
    addr = IR.reg(:u64, id)

    # Emitted on *every* access, not just the first. The register is reused,
    # but each pointer has a different base, so hoisting this would leave the
    # register holding the first pointer's address for all of them -- the
    # dropped-pointer-base bug this module already guards against once.
    scope =
      emit(scope, "add",
        dtype: :u64,
        dest: addr,
        ops: [IR.reg(:u64, base), IR.reg(:u64, scope.offset)]
      )

    {scope, IR.addr(addr)}
  end

  # Allocate the address register on first use, then reuse it. `alloc/2` numbers
  # it like any other register, so it shows up in the IR and in the ptxas
  # register count: nothing about it is hidden from the reader or the gate.
  defp materialize_addr(%Scope{addr_reg: nil} = scope) do
    {scope, id} = alloc(scope, :u64)
    {%{scope | addr_reg: id}, id}
  end

  defp materialize_addr(%Scope{addr_reg: id} = scope), do: {scope, id}

  defp alloc(%Scope{counter: counter} = scope, type) do
    class =
      IR.reg_class(type) || raise(ArgumentError, "no register class for type #{inspect(type)}")

    id = Map.get(counter, class, 0) + 1
    {%{scope | counter: Map.put(counter, class, id)}, id}
  end

  defp emit(%Scope{} = scope, base, opts) do
    %{scope | instrs: scope.instrs ++ [IR.instr(base, opts)]}
  end

  # `dest_type` decides the register class; `dtype` is what the opcode is annotated
  # with. They differ for `mul.wide` (writes `%rd`, reads and annotates `.u32`) and
  # for `setp`/`not`, whose destination is a predicate.
  defp emit_dest(%Scope{} = scope, base, dest_type, opts) do
    {scope, id} = alloc(scope, dest_type)
    dest = if dest_type == :pred, do: IR.pred(id), else: IR.reg(dest_type, id)
    {emit(scope, base, Keyword.put(opts, :dest, dest)), id}
  end

  defp operand_type({:reg, type, _id}), do: type
  defp operand_type({:immf, type, _value}), do: type
  defp operand_type({:imm, value}) when is_integer(value), do: :s32
end
