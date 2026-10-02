defmodule Gpark.BackendTest do
  @moduledoc """
  Tests for `Gpark.Backend`.

  The gate earns its place only if it refuses something. A `require!/2` that accepts
  every kernel is indistinguishable from no gate at all, so most of what follows builds
  a kernel the PTX backend genuinely cannot handle and checks that it is turned away
  with a specific reason rather than emitted.
  """

  use ExUnit.Case, async: true

  alias Gpark.Backend
  alias Gpark.IR

  @corpus [
    Gpark.Kernels.VecAddF32,
    Gpark.Kernels.SaxpyF32,
    Gpark.Kernels.ReduceSumF32,
    Gpark.Kernels.UnpackU4F32
  ]

  # A backend that knows nothing, used to prove the gate is doing the refusing rather
  # than the PTX backend happening to agree with itself.
  defmodule EmptyBackend do
    @behaviour Gpark.Backend

    @impl true
    def name, do: :empty

    @impl true
    def ops, do: ["ret"]

    @impl true
    def types, do: [:pred]

    @impl true
    def emit(_kernel), do: ""

    @impl true
    def check(_kernel), do: {:ok, %{}}
  end

  # A backend that emits the same text but claims one fewer opcode, which is the exact
  # shape of the real bug: a capability list that drifted from what the emitter does.
  defmodule NarrowedBackend do
    @behaviour Gpark.Backend

    @impl true
    def name, do: :narrowed

    @impl true
    def ops, do: Gpark.Ops.names() -- ["shfl"]

    @impl true
    def types, do: Gpark.Type.all()

    @impl true
    def emit(kernel), do: Gpark.PTX.emit(kernel)

    @impl true
    def check(kernel), do: Gpark.Validate.check(kernel)
  end

  defp ret, do: IR.instr("ret")

  defp one_instr_kernel(name, instr) do
    IR.kernel(name, blocks: [IR.block(:entry, [instr], ret())])
  end

  describe "the PTX backend's declared capabilities" do
    test "cover the whole ops table and the whole type set" do
      assert Gpark.PTX.ops() == Gpark.Ops.names()
      assert Gpark.PTX.types() == Gpark.Type.all()
      assert length(Gpark.PTX.ops()) == 48
      assert length(Gpark.PTX.types()) == 30
    end

    test "include the sub-byte types, which have no native PTX spelling" do
      # They are declared supported because they are representable: `ptx_type/1`
      # returns nil for a bare `:u4` because PTX cannot spell one, and the container
      # only appears once it is packed (`ptx_type(%Packed{container: :u32}) == :u32`).
      # Omitting them from the declared set would make require!/2 reject every
      # sub-byte kernel on a technicality.
      for type <- [:s2, :u2, :s4, :u4] do
        assert Backend.supports_type?(Gpark.PTX, type)
        refute Gpark.Type.native?(type), "#{type} must have no direct PTX spelling"
        assert Gpark.Type.ptx_type(type) == nil
        assert Gpark.Type.widen(type) in [:s16, :u16]
      end

      packed = Gpark.Type.packed(:u32, :u4, 8)
      assert Gpark.Type.ptx_type(packed) == :u32
    end

    test "has a name for diagnostics" do
      assert Gpark.PTX.name() == :ptx
    end
  end

  describe "required_ops/1 and required_types/1" do
    test "read opcodes from instructions and terminators alike" do
      # A ret-only body would make a terminator-blind implementation see no ops at all.
      kernel = IR.kernel(:t, blocks: [IR.block(:entry, [], ret())])
      assert Backend.required_ops(kernel) == MapSet.new(["ret"])
    end

    test "collect types from operands, dtypes and parameter declarations" do
      add = IR.instr("add", dtype: :u32, dest: {:reg, :u32, 1}, ops: [{:reg, :u32, 1}, {:imm, 1}])

      addr =
        IR.instr("ld",
          space: :global,
          dtype: :f32,
          dest: {:reg, :f32, 1},
          ops: [IR.addr({:reg, :u64, 2})]
        )

      st = IR.instr("st", space: :global, dtype: :f32, ops: [IR.addr({:reg, :u64, 2})])

      kernel =
        IR.kernel(:t,
          params: [IR.param_decl(:a, :u64)],
          blocks: [IR.block(:entry, [add, addr, st], ret())]
        )

      types = Backend.required_types(kernel)
      # u32 from the arithmetic, f32 from the load/store, u64 from the address base,
      # and u64 again from the parameter declaration.
      assert :u32 in types
      assert :f32 in types
      assert :u64 in types
    end

    test "do not conflate register ids across types" do
      add = IR.instr("add", dtype: :u32, dest: {:reg, :u32, 1}, ops: [{:reg, :u32, 1}, {:imm, 1}])

      kernel = one_instr_kernel(:t, add)
      types = Backend.required_types(kernel)

      assert :u32 in types
      refute :u64 in types
    end
  end

  describe "require!/2" do
    test "passes every corpus kernel" do
      for mod <- @corpus do
        built = mod.build()
        assert {:ok, kernel} = Backend.require!(Gpark.PTX, built)
        # The gate is a gate, not a transform: what comes back is what went in.
        assert kernel == built
        assert {:ok, _} = Gpark.PTX.check(kernel)
      end
    end

    test "returns the kernel unchanged" do
      # It is a gate, not a lowering step. A caller that gets {:ok, _} must be able to
      # assume nothing was rewritten on the way through.
      kernel = Gpark.Kernels.VecAddF32.build()
      assert {:ok, ^kernel} = Backend.require!(Gpark.PTX, kernel)
    end

    test "refuses an opcode the backend does not implement" do
      kernel = one_instr_kernel(:t, IR.instr("tensor::mma", dtype: :f32))

      assert {:error, [{:unsupported_op, "tensor::mma"}]} =
               Backend.require!(Gpark.PTX, kernel)
    end

    test "refuses a type the backend cannot represent" do
      kernel =
        one_instr_kernel(
          :t,
          IR.instr("add",
            dtype: :u128,
            dest: {:reg, :u128, 1},
            ops: [{:reg, :u128, 1}, {:imm, 1}]
          )
        )

      assert {:error, [{:unsupported_type, :u128}]} = Backend.require!(Gpark.PTX, kernel)
    end

    test "reports every missing capability at once rather than the first" do
      # Fixing one, re-running, and finding the next is a miserable loop.
      bad_op = IR.instr("tensor::mma", dtype: :f32)
      bad_type = IR.instr("cvt", dtype: :u128, dest: {:reg, :u128, 1}, ops: [{:reg, :f32, 1}])

      kernel = one_instr_kernel(:t, bad_op)
      kernel = %{kernel | blocks: [%{hd(kernel.blocks) | instrs: [bad_op, bad_type]}]}

      assert {:error, issues} = Backend.require!(Gpark.PTX, kernel)
      assert {:unsupported_op, "tensor::mma"} in issues
      assert {:unsupported_type, :u128} in issues
    end

    test "the gate is what refuses, not the backend agreeing with itself" do
      # Against EmptyBackend a plain corpus kernel must be rejected. If this passes
      # even with a backend supporting almost nothing, require!/2 is not checking.
      assert {:error, issues} = Backend.require!(EmptyBackend, Gpark.Kernels.VecAddF32.build())
      assert {:unsupported_op, "ld"} in issues
      assert {:unsupported_type, :u32} in issues
    end

    test "catches a capability list that drifted from the emitter" do
      # NarrowedBackend renders identical text to PTX but omits shfl. reduce_sum_f32
      # uses shfl, so it must be refused even though emitting it would have worked.
      assert {:ok, _} = Backend.require!(Gpark.PTX, Gpark.Kernels.ReduceSumF32.build())

      assert {:error, [{:unsupported_op, "shfl"}]} =
               Backend.require!(NarrowedBackend, Gpark.Kernels.ReduceSumF32.build())
    end
  end

  describe "emit!/2" do
    test "emits when the kernel is supported" do
      kernel = Gpark.Kernels.VecAddF32.build()
      assert Backend.emit!(Gpark.PTX, kernel) == Gpark.PTX.emit(kernel)
    end

    test "raises rather than emitting something unsupported" do
      kernel = one_instr_kernel(:t, IR.instr("tensor::mma", dtype: :f32))

      error =
        assert_raise ArgumentError, fn -> Backend.emit!(Gpark.PTX, kernel) end

      message = Exception.message(error)
      assert message =~ "cannot emit t"
      # The reason must name the capability, not just announce failure.
      assert message =~ "tensor::mma"
    end

    test "the raise message points at the two honest options" do
      kernel = one_instr_kernel(:t, IR.instr("tensor::mma", dtype: :f32))
      error = assert_raise ArgumentError, fn -> Backend.emit!(Gpark.PTX, kernel) end

      assert Exception.message(error) =~ "slower sequence"
      assert Exception.message(error) =~ "extend the backend"
    end
  end

  describe "check/1 versus require!/2" do
    test "a malformed kernel can pass require! and still fail check" do
      # The two answer different questions. A kernel using nothing exotic is within the
      # PTX backend's capabilities even when it is structurally broken, so conflating
      # them would hide the real error behind a capability message.
      st = IR.instr("st", space: :global, dtype: :f32, ops: [IR.addr({:reg, :u64, 1})])

      kernel = IR.kernel(:broken, blocks: [IR.block(:entry, [st], ret())])

      assert {:ok, _} = Backend.require!(Gpark.PTX, kernel)
      assert {:error, _issues} = Gpark.PTX.check(kernel)
    end
  end
end
