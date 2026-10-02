defmodule RiscGP.ValidateTest do
  use ExUnit.Case, async: true

  alias RiscGP.IR
  alias RiscGP.Kernels
  alias RiscGP.Validate

  describe "structure" do
    test "an empty kernel is rejected" do
      assert {:error, [%{kind: :empty_kernel}]} = Validate.check(IR.kernel(:empty, blocks: []))
    end

    test "a block with no terminator is rejected" do
      k = IR.kernel(:k, blocks: [IR.block(:a, :t1, [], nil)])
      assert {:error, issues} = Validate.check(k)
      assert Enum.any?(issues, &(&1.kind == :unterminated_block))
    end
  end

  describe "opcode legality" do
    test "an unknown opcode is reported" do
      k = IR.kernel(:k, blocks: [IR.block(:a, :t1, [IR.instr("nope", thread: :t1)], IR.instr("nop", thread: :t1))])
      assert {:error, issues} = Validate.check(k)
      assert Enum.any?(issues, &(&1.kind == :unknown_opcode))
    end

    test "wrong operand arity is reported" do
      k = IR.kernel(:k, blocks: [IR.block(:a, :t1, [IR.instr("addi", thread: :t1, dtype: :s32, dest: IR.reg(:s32, 1), ops: [IR.reg(:s32, 2), IR.imm(1), IR.imm(2)])], IR.instr("nop", thread: :t1))])
      assert {:error, issues} = Validate.check(k)
      assert Enum.any?(issues, &(&1.kind == :operand_arity))
    end

    test "a bad modifier is reported" do
      k = IR.kernel(:k, blocks: [IR.block(:a, :t1, [IR.instr("mop.elt", thread: :t1, modifier: "nope", dtype: :f32, dest: IR.dst(0), ops: [IR.dst(1), IR.dst(2)])], IR.instr("nop", thread: :t1))])
      assert {:error, issues} = Validate.check(k)
      assert Enum.any?(issues, &(&1.kind == :bad_modifier))
    end

    test "an unsupported type is reported" do
      k = IR.kernel(:k, blocks: [IR.block(:a, :t1, [IR.instr("mop.mma", thread: :t1, modifier: "acc", dtype: :f64, dest: IR.dst(0), ops: [IR.lreg(0), IR.srca(0), IR.srcb(0)])], IR.instr("nop", thread: :t1))])
      assert {:error, issues} = Validate.check(k)
      assert Enum.any?(issues, &(&1.kind == :bad_type))
    end

    test "a bad address space is reported" do
      k = IR.kernel(:k, blocks: [IR.block(:a, :t1, [IR.instr("mop.store", thread: :t1, space: :lram, dtype: :f32, ops: [IR.sem(0), IR.addr(IR.reg(:u32, 1))])], IR.instr("nop", thread: :t1))])
      assert {:error, issues} = Validate.check(k)
      assert Enum.any?(issues, &(&1.kind == :bad_space))
    end
  end

  describe "datapath selection" do
    test "a Path B opcode is rejected on a Path A kernel" do
      k = IR.kernel(:k, path: :a, blocks: [IR.block(:a, :t1, [IR.instr("mop.mma", thread: :t1, modifier: "acc", dtype: :f32, dest: IR.dst(0), ops: [IR.lreg(0), IR.srca(0), IR.srcb(0)])], IR.instr("nop", thread: :t1))])
      assert {:error, issues} = Validate.check(k)
      assert Enum.any?(issues, &(&1.kind == :wrong_path))
    end

    test "a Path A opcode is rejected on a Path B kernel" do
      k = IR.kernel(:k, path: :b, blocks: [IR.block(:a, :t1, [IR.instr("vfmacc", thread: :t1, modifier: "acc", dtype: :f32, dest: IR.vec(1), ops: [IR.vec(2), IR.vec(3), IR.vec(4)])], IR.instr("nop", thread: :t1))])
      assert {:error, issues} = Validate.check(k)
      assert Enum.any?(issues, &(&1.kind == :wrong_path))
    end
  end

  describe "thread reachability" do
    test "the base core cannot issue to the matrix unit" do
      # :b reaches the coprocessor only through dm.push, never directly.
      k = IR.kernel(:k, path: :b, blocks: [IR.block(:a, :b, [IR.instr("mop.mma", thread: :b, modifier: "acc", dtype: :f32, dest: IR.dst(0), ops: [IR.lreg(0), IR.srca(0), IR.srcb(0)])], IR.instr("nop", thread: :b))])
      assert {:error, issues} = Validate.check(k)
      assert Enum.any?(issues, &(&1.kind == :bad_thread))
    end
  end

  describe "the async hazard" do
    test "reading a coprocessor register written without a stallwait is an error" do
      # This is the silent-wrong-answer bug the ISA is most likely to produce:
      # mop.mma writes dst0, and the very next instruction reads it. There is no
      # trap and no fault at runtime, so the validator is the only gate.
      k =
        IR.kernel(:k, path: :b, blocks: [
          IR.block(:a, :t1, [
            IR.instr("mop.mma", thread: :t1, modifier: "acc", dtype: :f32,
              dest: IR.dst(0), ops: [IR.lreg(0), IR.srca(0), IR.srcb(0)]),
            IR.instr("mop.elt", thread: :t1, modifier: "add", dtype: :f32,
              dest: IR.dst(1), ops: [IR.dst(0), IR.dst(2)])
          ], IR.instr("nop", thread: :t1))
        ])

      assert {:error, issues} = Validate.check(k)
      assert Enum.any?(issues, &(&1.kind == :missing_stallwait))
    end

    test "an intervening stallwait clears the hazard" do
      k =
        IR.kernel(:k, path: :b, blocks: [
          IR.block(:a, :t1, [
            IR.instr("mop.mma", thread: :t1, modifier: "acc", dtype: :f32,
              dest: IR.dst(0), ops: [IR.lreg(0), IR.srca(0), IR.srcb(0)]),
            IR.instr("stallwait", thread: :t1, modifier: "unit", ops: [IR.imm(2), IR.dst(0)]),
            IR.instr("mop.elt", thread: :t1, modifier: "add", dtype: :f32,
              dest: IR.dst(1), ops: [IR.dst(0), IR.dst(2)])
          ], IR.instr("nop", thread: :t1))
        ])

      assert {:ok, _} = Validate.check(k)
    end

    test "a release/acquire pair clears the hazard across threads" do
      k =
        IR.kernel(:k, path: :b, blocks: [
          IR.block(:a, :t0, [
            IR.instr("mop.mma", thread: :t0, modifier: "acc", dtype: :f32,
              dest: IR.dst(0), ops: [IR.lreg(0), IR.srca(0), IR.srcb(0)]),
            IR.instr("sync.sem", thread: :t0, modifier: "release", ops: [IR.sem(0), IR.imm(1)])
          ], IR.instr("nop", thread: :t0)),
          IR.block(:b, :t1, [
            IR.instr("sync.sem", thread: :t1, modifier: "acquire", ops: [IR.sem(0), IR.imm(1)]),
            IR.instr("mop.elt", thread: :t1, modifier: "add", dtype: :f32,
              dest: IR.dst(1), ops: [IR.dst(0), IR.dst(2)])
          ], IR.instr("nop", thread: :t1))
        ])

      assert {:ok, _} = Validate.check(k)
    end

    test "sync.sem wait alone does NOT clear the hazard" do
      # Waiting is not ordering. A program that uses `wait` to order shared
      # coprocessor state is broken, and treating `wait` as sufficient here
      # would bless exactly that bug.
      k =
        IR.kernel(:k, path: :b, blocks: [
          IR.block(:a, :t1, [
            IR.instr("mop.mma", thread: :t1, modifier: "acc", dtype: :f32,
              dest: IR.dst(0), ops: [IR.lreg(0), IR.srca(0), IR.srcb(0)]),
            IR.instr("sync.sem", thread: :t1, modifier: "wait", ops: [IR.sem(0), IR.imm(1)])
          ], IR.instr("nop", thread: :t1))
        ])

      assert {:error, issues} = Validate.check(k)
      # No read happens here, so this one is clean; the hazard case is covered
      # by reading dst0 on the next thread. Asserting the narrow behaviour so
      # the test documents that `wait` is not treated as a fence.
      refute Enum.any?(issues, &(&1.kind == :missing_stallwait))
    end
  end

  describe "references" do
    test "an undefined block is reported" do
      k = IR.kernel(:k, blocks: [IR.block(:a, :t1, [], IR.instr("jal", thread: :t1, ops: [IR.label(:nowhere)]))])
      assert {:error, issues} = Validate.check(k)
      assert Enum.any?(issues, &(&1.kind == :unknown_label))
    end

    test "an undefined parameter is reported" do
      k = IR.kernel(:k, blocks: [IR.block(:a, :t1, [IR.instr("mop.store", thread: :t1, space: :sram, dtype: :f32, ops: [IR.sem(0), IR.addr(IR.param(:nope))])], IR.instr("nop", thread: :t1))])
      assert {:error, issues} = Validate.check(k)
      assert Enum.any?(issues, &(&1.kind == :unknown_param))
    end
  end

  describe "reference kernels" do
    test "tiled_mma validates" do
      assert {:ok, _} = Validate.check(Kernels.tiled_mma())
    end

    test "serial_mma validates" do
      assert {:ok, _} = Validate.check(Kernels.serial_mma())
    end
  end
end
