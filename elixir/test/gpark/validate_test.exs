defmodule Gpark.ValidateTest do
  use ExUnit.Case, async: true

  alias Gpark.IR
  alias Gpark.Validate

  defp issues_for(kernel) do
    case Validate.check(kernel) do
      {:ok, _} -> []
      {:error, issues} -> Enum.map(issues, & &1.kind)
    end
  end

  defp kernel_with(instrs, term) do
    IR.kernel(:t, params: [IR.param_decl(:x, :u64)], blocks: [IR.block(:entry, instrs, term)])
  end

  describe "opcode legality" do
    test "rejects an opcode outside the table" do
      k = kernel_with([IR.instr("frobnicate", dtype: :u32)], IR.instr("ret"))
      assert :unknown_opcode in issues_for(k)
    end

    test "rejects a type the opcode does not support" do
      k =
        kernel_with(
          [IR.instr("rsqrt", dtype: :f64, dest: IR.reg(:f64, 1), ops: [IR.immf(:f64, 1.0)])],
          IR.instr("ret")
        )

      assert :bad_type in issues_for(k)
    end

    test "accepts a legal opcode" do
      k =
        kernel_with(
          [IR.instr("rsqrt", dtype: :f32, dest: IR.reg(:f32, 1), ops: [IR.immf(:f32, 2.0)])],
          IR.instr("ret")
        )

      assert issues_for(k) == []
    end
  end

  describe "arity" do
    test "rejects the wrong operand count" do
      k =
        kernel_with(
          [IR.instr("add", dtype: :u32, dest: IR.reg(:u32, 1), ops: [IR.imm(1)])],
          IR.instr("ret")
        )

      assert :operand_arity in issues_for(k)
    end

    test "rejects a destination on a store" do
      k =
        kernel_with(
          [
            IR.instr("st",
              dtype: :u32,
              space: :global,
              dest: IR.reg(:u32, 1),
              ops: [IR.addr(IR.param(:x)), IR.imm(0)]
            )
          ],
          IR.instr("ret")
        )

      assert :dest_arity in issues_for(k)
    end
  end

  describe "references" do
    test "rejects a branch to a block that does not exist" do
      k = kernel_with([], IR.instr("bra", ops: [IR.label(:nowhere)]))
      assert :unknown_label in issues_for(k)
    end

    test "rejects a parameter that is not declared" do
      k =
        kernel_with(
          [
            IR.instr("ld",
              dtype: :u64,
              space: :param,
              dest: IR.reg(:u64, 1),
              ops: [IR.param(:nope)]
            )
          ],
          IR.instr("ret")
        )

      assert :unknown_param in issues_for(k)
    end
  end

  describe "register typing" do
    test "accepts the same register id in different banks" do
      # %f1 and %r1 are different registers even though both are "1".
      instrs = [
        IR.instr("mov", dtype: :u32, dest: IR.reg(:u32, 1), ops: [IR.imm(1)]),
        IR.instr("mov", dtype: :f32, dest: IR.reg(:f32, 1), ops: [IR.immf(:f32, 1.0)])
      ]

      assert issues_for(kernel_with(instrs, IR.instr("ret"))) == []
    end

    test "rejects one register id used at two types in the same bank" do
      instrs = [
        IR.instr("mov", dtype: :u32, dest: IR.reg(:u32, 1), ops: [IR.imm(1)]),
        IR.instr("mov", dtype: :s32, dest: IR.reg(:s32, 1), ops: [IR.imm(1)])
      ]

      assert :register_type_conflict in issues_for(kernel_with(instrs, IR.instr("ret")))
    end
  end

  describe "initialisation" do
    test "rejects reading a register that was never written" do
      instrs = [
        IR.instr("add", dtype: :u32, dest: IR.reg(:u32, 1), ops: [IR.reg(:u32, 9), IR.imm(1)])
      ]

      assert :uninitialised_register in issues_for(kernel_with(instrs, IR.instr("ret")))
    end

    test "accepts a read after a write" do
      instrs = [
        IR.instr("mov", dtype: :u32, dest: IR.reg(:u32, 9), ops: [IR.imm(1)]),
        IR.instr("add", dtype: :u32, dest: IR.reg(:u32, 1), ops: [IR.reg(:u32, 9), IR.imm(1)])
      ]

      assert issues_for(kernel_with(instrs, IR.instr("ret"))) == []
    end

    test "writes in an earlier block count as initialising" do
      blocks = [
        IR.block(
          :entry,
          [IR.instr("mov", dtype: :u32, dest: IR.reg(:u32, 1), ops: [IR.imm(1)])],
          IR.instr("bra", ops: [IR.label(:second)])
        ),
        IR.block(
          :second,
          [IR.instr("ret")],
          IR.instr("add", dtype: :u32, dest: IR.reg(:u32, 2), ops: [IR.reg(:u32, 1), IR.imm(1)])
        )
      ]

      assert issues_for(IR.kernel(:t, params: [], blocks: blocks)) == []
    end
  end

  describe "structure" do
    test "rejects a block with no terminator" do
      k = IR.kernel(:t, params: [], blocks: [IR.block(:entry, [IR.instr("nop")], nil)])
      assert :unterminated_block in issues_for(k)
    end

    test "rejects an empty kernel" do
      assert :empty_kernel in issues_for(IR.kernel(:t, params: [], blocks: []))
    end

    test "rejects duplicate block labels" do
      k =
        IR.kernel(:t,
          params: [],
          blocks: [
            IR.block(:entry, [], IR.instr("bra", ops: [IR.label(:entry)])),
            IR.block(:entry, [], IR.instr("ret"))
          ]
        )

      assert :duplicate_block in issues_for(k)
    end
  end

  describe "exit counting" do
    # `note_exit/2` used to issue :multiple_exits on the *first* `exit`, which meant
    # every kernel with any exit at all was flagged -- a check that always fires and
    # therefore gets ignored. The first exit is how a kernel legitimately returns;
    # only later ones are redundant.
    test "a single exit is not reported" do
      k = kernel_with([], IR.instr("exit"))
      refute :multiple_exits in issues_for(k)
    end

    test "a second exit is reported" do
      blocks = [
        IR.block(:a, [], IR.instr("exit")),
        IR.block(:b, [], IR.instr("exit"))
      ]

      k = IR.kernel(:t, params: [IR.param_decl(:x, :u64)], blocks: blocks)
      assert :multiple_exits in issues_for(k)
    end

    test "multiple rets are not exits" do
      # An early-return block plus a `done` block legitimately contains several
      # `ret`s; PTX is perfectly happy with that.
      blocks = [
        IR.block(:entry, [], IR.instr("ret")),
        IR.block(:done, [], IR.instr("ret"))
      ]

      k = IR.kernel(:t, params: [IR.param_decl(:x, :u64)], blocks: blocks)
      refute :multiple_exits in issues_for(k)
    end
  end

  describe "signedness" do
    # These both delegated to Type.int?/1, so signed_type? answered "true" for
    # :u32 and unsigned_type? answered "true" for :i32. Sign is what tells a
    # comparison instruction which predicate modifier to emit, so a caller relying
    # on either would have generated a wrapped-vs-signed comparison.
    test "distinguishes signed from unsigned" do
      assert IR.signed_type?(:s32)
      assert IR.signed_type?(:s64)
      refute IR.signed_type?(:u32)

      assert IR.unsigned_type?(:u32)
      assert IR.unsigned_type?(:u64)
      refute IR.unsigned_type?(:s32)
    end

    test "rejects non-integers" do
      refute IR.signed_type?(:f32)
      refute IR.unsigned_type?(:f32)
      refute IR.signed_type?(:pred)
    end

    test "resolves packed sub-byte types" do
      # Type.sign/1 recurses through the container, so a packed s4 is signed even
      # though its storage is a b32.
      signed = Gpark.Type.packed(:b32, :s4, 8)
      unsigned = Gpark.Type.packed(:b32, :u4, 8)

      assert IR.signed_type?(signed)
      refute IR.unsigned_type?(signed)
      assert IR.unsigned_type?(unsigned)
      refute IR.signed_type?(unsigned)
    end
  end

  describe "reporting" do
    test "reports every problem, not just the first" do
      instrs = [
        IR.instr("frobnicate", dtype: :u32),
        IR.instr("add", dtype: :u32, dest: IR.reg(:u32, 1), ops: [IR.imm(1)])
      ]

      assert length(issues_for(kernel_with(instrs, IR.instr("ret")))) >= 2
    end

    test "check_all/1 aggregates across kernels" do
      assert :ok = Validate.check_all([Gpark.Kernels.VecAddF32.build()])
    end
  end
end
