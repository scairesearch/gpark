defmodule Gpark.SimplifyTest do
  @moduledoc """
  Tests for `Gpark.Opt.Simplify`.

  The corpus tests establish that simplification does not disturb the goldens. They
  cannot establish that the rules work, because every rule reports zero on already
  minimal kernels -- a pass that fires nothing and a pass that is broken look
  identical. The synthetic kernels below exist to give each rule something to
  actually remove, and to pin the cases where removing something would be a bug.
  """

  use ExUnit.Case, async: true

  alias Gpark.IR
  alias Gpark.Opt.Simplify

  @corpus [
    Gpark.Kernels.VecAddF32,
    Gpark.Kernels.SaxpyF32,
    Gpark.Kernels.ReduceSumF32,
    Gpark.Kernels.UnpackU4F32
  ]

  defp bases(%{blocks: blocks}) do
    Enum.flat_map(blocks, & &1.instrs) |> Enum.map(& &1.base)
  end

  defp labels(%{blocks: blocks}), do: Enum.map(blocks, & &1.label)

  defp ret, do: IR.instr("ret")
  defp bra(label), do: IR.instr("bra", ops: [{:label, label}])

  # ===========================================================================
  # Opcode classification
  # ===========================================================================

  describe "opcode classification" do
    test "every opcode in the table is classified" do
      unclassified = Gpark.Ops.names() -- (Simplify.pure_ops() ++ Simplify.impure_ops())
      assert unclassified == [], "unclassified opcodes default to impure on purpose, so add them"
    end

    test "pure and impure lists do not overlap" do
      overlap =
        MapSet.intersection(
          MapSet.new(Simplify.pure_ops()),
          MapSet.new(Simplify.impure_ops())
        )

      assert Enum.to_list(overlap) == []

      combined = Simplify.pure_ops() ++ Simplify.impure_ops()
      assert Enum.uniq(combined) == combined, "an opcode is classified twice"
    end

    test "an unknown opcode is :unknown rather than :pure" do
      # The failure mode this guards: a new opcode nobody classified being treated as
      # pure, so the pass deletes an instruction that writes memory. `classify/1` takes
      # a base name -- the IR keeps space, modifier and vector width in separate fields
      # -- so a fully-spelled opcode is not something it will ever see, and lands on
      # :unknown, which callers must treat as impure.
      assert Simplify.classify("red") == :impure
      assert Simplify.classify("red.global.add.s32") == :unknown
      assert Simplify.pure?("red.global.add.s32") == false
      assert Simplify.pure?("not.an.opcode") == false
    end

    test "memory, control flow and synchronisation opcodes are impure" do
      for base <- ~w(st ld prefetch bra brx ret exit bar shfl vote activemask atom red) do
        assert Simplify.pure?(base) == false, "#{base} must not be removable"
      end
    end

    test "arithmetic opcodes are pure" do
      for base <- ~w(add sub mul fma mad mad.lo neg not and or xor shl shr setp selp) do
        assert Simplify.pure?(base) == true, "#{base} should be removable"
      end
    end
  end

  # ===========================================================================
  # The corpus: simplification must not disturb the goldens
  # ===========================================================================

  describe "corpus" do
    test "emitted PTX is byte-identical before and after simplification" do
      for mod <- @corpus do
        kernel = mod.build()

        assert Gpark.PTX.emit(Simplify.simplify(kernel)) == Gpark.PTX.emit(kernel),
               "#{kernel.name}: simplification rewrote bytes"
      end
    end

    test "simplified kernels still validate" do
      for mod <- @corpus do
        assert {:ok, _kernel} = Gpark.Validate.check(Simplify.simplify(mod.build()))
      end
    end

    test "simplify is idempotent" do
      for mod <- @corpus do
        once = Simplify.simplify(mod.build())
        assert Simplify.simplify(once) == once
      end
    end

    test "the corpus is already fixpoint-minimal" do
      # Zero iterations and zero removals. This is the claim that the hand-written
      # kernels have no dead code and no unreachable blocks -- if a future kernel edit
      # introduces either, this test is what says so.
      for mod <- @corpus do
        {_kernel, stats} = Simplify.simplify(mod.build(), [])

        assert stats.iterations == 0, "#{mod.build().name} is not minimal"

        assert Enum.all?(Map.values(stats.rules), &(&1 == 0)),
               "#{mod.build().name}: #{inspect(stats.rules)}"
      end
    end

    test "changed? is false for every corpus kernel" do
      for mod <- @corpus, do: refute(Simplify.changed?(mod.build()))
    end

    test "stats report every rule, including ones that never fire" do
      {_kernel, stats} = Simplify.simplify(Gpark.Kernels.VecAddF32.build(), [])

      assert Map.keys(stats.rules) == [:drop_unreachable_blocks, :drop_dead_instructions]
      assert is_integer(stats.iterations)
    end
  end

  # ===========================================================================
  # Rule: unreachable blocks
  # ===========================================================================

  describe "drop_unreachable_blocks" do
    test "removes a block nothing branches to" do
      entry = IR.block(:entry, [], bra(:done))
      done = IR.block(:done, [], ret())
      orphan = IR.block(:orphan, [], ret())

      kernel = IR.kernel(:synthetic, blocks: [entry, done, orphan])
      {simplified, stats} = Simplify.simplify(kernel, [])

      assert labels(simplified) == [:entry, :done]
      assert stats.rules[:drop_unreachable_blocks] == 1
    end

    test "keeps a block reached by fallthrough from a predicated branch" do
      setp =
        IR.instr("setp",
          modifier: "ge",
          dtype: :u32,
          dest: {:pred, 1},
          ops: [{:reg, :u32, 1}, {:imm, 4}]
        )

      entry =
        IR.block(:entry, [setp], IR.instr("bra", ops: [{:label, :done}], pred: {:pred, 1}))

      fallthrough = IR.block(:body, [], ret())
      done = IR.block(:done, [], ret())

      kernel = IR.kernel(:synthetic, blocks: [entry, fallthrough, done])
      simplified = Simplify.simplify(kernel)

      assert labels(simplified) == [:entry, :body, :done]
    end

    test "removes the block after an unconditional ret" do
      entry = IR.block(:entry, [], ret())
      dead_tail = IR.block(:tail, [], ret())

      kernel = IR.kernel(:synthetic, blocks: [entry, dead_tail])
      simplified = Simplify.simplify(kernel)

      assert labels(simplified) == [:entry]
    end

    test "removes the block after an unconditional branch, not just the untaken one" do
      entry = IR.block(:entry, [], bra(:done))
      unreachable_middle = IR.block(:middle, [], ret())
      done = IR.block(:done, [], ret())

      kernel = IR.kernel(:synthetic, blocks: [entry, unreachable_middle, done])
      simplified = Simplify.simplify(kernel)

      assert labels(simplified) == [:entry, :done]
    end
  end

  # ===========================================================================
  # Rule: dead instructions
  # ===========================================================================

  describe "drop_dead_instructions" do
    test "removes a pure instruction whose destination is never read" do
      dead =
        IR.instr("add", dtype: :u32, dest: {:reg, :u32, 9}, ops: [{:reg, :u32, 1}, {:imm, 1}])

      kernel = IR.kernel(:synthetic, blocks: [IR.block(:entry, [dead], ret())])
      {simplified, stats} = Simplify.simplify(kernel, [])

      refute "add" in bases(simplified)
      assert stats.rules[:drop_dead_instructions] == 1
    end

    test "keeps a pure instruction whose destination is read" do
      live =
        IR.instr("add", dtype: :u32, dest: {:reg, :u32, 9}, ops: [{:reg, :u32, 1}, {:imm, 1}])

      consumer =
        IR.instr("mul", dtype: :u32, dest: {:reg, :u32, 10}, ops: [{:reg, :u32, 9}, {:imm, 2}])

      st = IR.instr("st", space: :global, vec: nil, dtype: :u32, ops: [IR.addr({:reg, :u32, 10})])

      kernel = IR.kernel(:synthetic, blocks: [IR.block(:entry, [live, consumer, st], ret())])
      simplified = Simplify.simplify(kernel)

      assert bases(simplified) == ["add", "mul", "st"]
    end

    test "keeps a dead load, because removing it could hide a fault" do
      # The regression this whole policy exists for: unpack_u4_f32 once emitted its
      # bounds guard after the work it guarded, and a dead-load rule would have
      # deleted the evidence.
      load =
        IR.instr("ld",
          space: :global,
          dtype: :u32,
          ops: [IR.addr({:reg, :u64, 1})],
          dest: {:reg, :u32, 7}
        )

      kernel = IR.kernel(:synthetic, blocks: [IR.block(:entry, [load], ret())])
      simplified = Simplify.simplify(kernel)

      assert bases(simplified) == ["ld"]
    end

    test "keeps stores, which have no destination to test" do
      st = IR.instr("st", space: :global, dtype: :u32, ops: [IR.addr({:reg, :u32, 1})])

      kernel = IR.kernel(:synthetic, blocks: [IR.block(:entry, [st], ret())])
      simplified = Simplify.simplify(kernel)

      assert bases(simplified) == ["st"]
    end

    test "keeps a dead shuffle and barrier" do
      shuffle =
        IR.instr("shfl",
          dtype: :u32,
          dest: {:reg, :u32, 3},
          ops: [{:reg, :u32, 1}, {:reg, :u32, 2}]
        )

      barrier = IR.instr("bar", ops: [{:imm, 0}])

      kernel = IR.kernel(:synthetic, blocks: [IR.block(:entry, [shuffle, barrier], ret())])
      simplified = Simplify.simplify(kernel)

      assert bases(simplified) == ["shfl", "bar"]
    end

    test "removes a dead predicate computation" do
      setp =
        IR.instr("setp",
          modifier: "lt",
          dtype: :u32,
          dest: {:pred, 1},
          ops: [{:reg, :u32, 1}, {:imm, 4}]
        )

      notp = IR.instr("not", dtype: :pred, dest: {:pred, 2}, ops: [{:pred, 1}])

      kernel = IR.kernel(:synthetic, blocks: [IR.block(:entry, [setp, notp], ret())])
      simplified = Simplify.simplify(kernel)

      assert bases(simplified) == []
    end

    test "does not conflate registers that share a numeric id across types" do
      # gpark reuses numeric register ids per type: u32 1 and u64 1 are different
      # registers. Keying liveness on a bare id would delete a live u32 write because
      # an unrelated u64 register with the same number is read.
      dead_u32 =
        IR.instr("add", dtype: :u32, dest: {:reg, :u32, 1}, ops: [{:reg, :u32, 5}, {:imm, 1}])

      read_u64 = IR.instr("mov", dtype: :u64, dest: {:reg, :u64, 2}, ops: [{:reg, :u64, 1}])

      st = IR.instr("st", space: :global, dtype: :u64, ops: [IR.addr({:reg, :u64, 2})])

      kernel =
        IR.kernel(:synthetic, blocks: [IR.block(:entry, [dead_u32, read_u64, st], ret())])

      simplified = Simplify.simplify(kernel)

      # The u32 write is genuinely dead and may go; the u64 chain must survive intact.
      assert bases(simplified) == ["mov", "st"]
    end

    test "removes a dead instruction in a later block" do
      entry = IR.block(:entry, [], bra(:body))

      dead =
        IR.instr("add", dtype: :u32, dest: {:reg, :u32, 9}, ops: [{:reg, :u32, 1}, {:imm, 1}])

      body = IR.block(:body, [dead], ret())

      kernel = IR.kernel(:synthetic, blocks: [entry, body])
      simplified = Simplify.simplify(kernel)

      assert bases(simplified) == []
    end
  end

  # ===========================================================================
  # Fixpoint behaviour
  # ===========================================================================

  describe "fixpoint" do
    test "keeps going until no rule fires, not for a fixed number of passes" do
      # The only reader of %r_u32_1 lives in a block nothing reaches. One pass removes
      # the block; only then does the write become dead. A single-pass pass would
      # leave it behind, and a bounded pass count would leave it behind too whenever
      # the chain got deeper than the bound.
      dead =
        IR.instr("add", dtype: :u32, dest: {:reg, :u32, 1}, ops: [{:reg, :u32, 5}, {:imm, 1}])

      reader = IR.instr("mov", dtype: :u32, dest: {:reg, :u32, 6}, ops: [{:reg, :u32, 1}])

      entry = IR.block(:entry, [dead], ret())
      orphan = IR.block(:orphan, [reader], ret())

      kernel = IR.kernel(:synthetic, blocks: [entry, orphan])
      {simplified, stats} = Simplify.simplify(kernel, [])

      assert stats.iterations >= 2, "must iterate: pass 1 drops the block, pass 2 the write"
      assert stats.rules[:drop_unreachable_blocks] == 1
      assert stats.rules[:drop_dead_instructions] == 1
      assert bases(simplified) == []
    end

    test "converges on a chain of dead instructions several rounds deep" do
      # t3 reads t2 reads t1. Backward liveness sees none of them as live, so a
      # single round clears the chain; this checks the pass does not get stuck or
      # corrupt it when the dependency runs backwards through the block.
      t1 = IR.instr("add", dtype: :u32, dest: {:reg, :u32, 1}, ops: [{:reg, :u32, 5}, {:imm, 1}])
      t2 = IR.instr("add", dtype: :u32, dest: {:reg, :u32, 2}, ops: [{:reg, :u32, 1}, {:imm, 1}])
      t3 = IR.instr("add", dtype: :u32, dest: {:reg, :u32, 3}, ops: [{:reg, :u32, 2}, {:imm, 1}])

      kernel = IR.kernel(:synthetic, blocks: [IR.block(:entry, [t1, t2, t3], ret())])
      simplified = Simplify.simplify(kernel)

      assert bases(simplified) == []
    end

    test "an empty kernel is left alone" do
      kernel = IR.kernel(:empty, blocks: [])
      assert Simplify.simplify(kernel) == kernel
    end
  end
end
