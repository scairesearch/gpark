defmodule Gpark.DSLTest do
  @moduledoc """
  The DSL's correctness argument is that it reproduces a kernel a careful human wrote
  by hand. These tests check that claim rather than asserting the DSL is
  self-consistent, which would prove nothing.
  """

  use ExUnit.Case, async: true

  import Gpark.DSL

  alias Gpark.Backend
  alias Gpark.DSL
  alias Gpark.IR
  alias Gpark.Kernels.VecAddF32
  alias Gpark.PTX
  alias Gpark.Validate

  defp dsl_vec_add do
    DSL.elementwise(:vec_add_f32,
      pointers: [a: :f32, b: :f32, out: :f32],
      count: :n,
      index: :ctaid_x,
      body: fn k ->
        {k, a} = load(k, :a)
        {k, b} = load(k, :b)
        {k, sum} = op(k, "add", :f32, [a, b])
        store(k, :out, sum)
      end
    )
  end

  describe "matching a hand-written kernel" do
    test "reproduces the hand-written IR exactly" do
      assert dsl_vec_add() == VecAddF32.build()
    end

    test "reproduces the golden PTX byte for byte" do
      golden =
        File.read!(Path.join([__DIR__, "..", "..", "..", "corpus", "golden", "vec_add_f32.ptx"]))

      assert PTX.emit(dsl_vec_add()) == golden
    end
  end

  describe "the result is a normal kernel" do
    test "passes validation" do
      assert {:ok, _} = Validate.check(dsl_vec_add())
    end

    test "passes the backend gate" do
      assert {:ok, _} = Backend.require!(PTX, dsl_vec_add())
    end

    test "emits through the backend gate like any other kernel" do
      assert Backend.emit!(PTX, dsl_vec_add()) =~ "vec_add_f32"
    end

    test "survives the simplifier with byte-identical output" do
      kernel = dsl_vec_add()
      simplified = Gpark.Opt.Simplify.simplify(kernel)
      assert simplified == kernel
      assert PTX.emit(simplified) == PTX.emit(kernel)
    end

    test "round-trips through the JSON corpus codec" do
      json = IR.JSON.encode!(dsl_vec_add())
      assert IR.JSON.decode!(json) == dsl_vec_add()
    end
  end

  describe "register allocation" do
    test "numbers per register class in first-definition order" do
      kernel = dsl_vec_add()

      # %rd: a, b, out, the byte offset, then the materialised address.
      # %r: n, then the index. %p: the guard and its negation.
      # %f: a, b, sum.
      assert Gpark.IR.max_regs(Enum.flat_map(kernel.blocks, & &1.instrs)) == %{
               rd: 5,
               r: 2,
               p: 2,
               f: 3
             }
    end

    test "never reuses a register within a kernel, except the address scratch" do
      instrs = dsl_vec_add() |> Map.fetch!(:blocks) |> Enum.flat_map(& &1.instrs)

      # The one deliberate reuse. PTX has no base+register addressing mode, so
      # every load and store needs its address in a register first, and the
      # addresses are not live at the same time. This is still visible in the
      # IR and counted by ptxas -- it is a reserved scratch, not an allocator
      # quietly inventing registers behind the reader's back.
      scratch =
        instrs
        |> Enum.filter(&(&1.base == "add" and &1.dtype == :u64))
        |> Enum.map(& &1.dest)
        |> Enum.uniq()
        |> Enum.map(fn {:reg, type, id} -> {IR.reg_class(type), id} end)
        |> Enum.uniq()

      assert scratch != []

      written =
        instrs
        |> Enum.map(& &1.dest)
        |> Enum.reject(&is_nil/1)

      regs =
        Enum.map(written, fn
          # %rd1 and %r1 are different registers in different banks, so the class
          # has to be part of the identity. Comparing bare ids would flag every
          # kernel as reusing registers.
          {:reg, type, id} -> {IR.reg_class(type), id}
          {:pred, id} -> {:p, id}
        end)

      # Value registers are strictly single-assignment. The only repeat allowed
      # anywhere is the address scratch, and it must be the sole exception --
      # if anything else starts repeating, that is a bug, not a policy.
      reused = regs -- Enum.uniq(regs)

      assert Enum.uniq(reused) == scratch

      # It is rewritten once per access (three here: two loads and a store), and
      # not once per kernel, so the reuse is bounded and visible rather than a
      # register being handed around invisibly.
      accesses =
        instrs
        |> Enum.filter(&(&1.base in ["ld", "st"] and &1.space == :global))
        |> Enum.flat_map(& &1.ops)
        |> Enum.filter(&match?({:addr, _, _, _}, &1))
        |> length()

      assert Enum.count(regs, &(&1 == hd(scratch))) == accesses
    end
  end

  describe "configuration" do
    test "honours a different index register" do
      kernel =
        DSL.elementwise(:k,
          pointers: [out: :f32],
          count: :n,
          index: :tid_x,
          body: fn k ->
            {k, v} = load(k, :out)
            store(k, :out, v)
          end
        )

      ops = Enum.flat_map(kernel.blocks, & &1.instrs) |> Enum.flat_map(& &1.ops)
      assert {:sreg, :tid_x} in ops

      assert {:ok, _} = Validate.check(kernel)
    end

    test "honours a custom guard label" do
      kernel =
        DSL.elementwise(:k,
          pointers: [out: :f32],
          count: :n,
          guard_label: :finish,
          body: fn k ->
            {k, v} = load(k, :out)
            store(k, :out, v)
          end
        )

      assert Enum.map(kernel.blocks, & &1.label) == [:entry, :finish]
      assert {:ok, _} = Validate.check(kernel)
    end

    test "uses the element width of the first pointer for the byte offset" do
      kernel =
        DSL.elementwise(:k,
          pointers: [a: :f64, out: :f64],
          count: :n,
          body: fn k ->
            {k, v} = load(k, :a)
            store(k, :out, v)
          end
        )

      mul = Enum.find(kernel.blocks |> hd() |> Map.fetch!(:instrs), &(&1.base == "mul.wide"))
      assert div(IR.width(:f64), 8) == 8
      assert {:imm, 8} in mul.ops
    end
  end

  describe "errors" do
    test "refuses to load from an undeclared pointer" do
      assert_raise ArgumentError, ~r/no pointer :nope declared/, fn ->
        DSL.elementwise(:k,
          pointers: [a: :f32],
          count: :n,
          body: fn k -> elem(load(k, :nope), 1) end
        )
      end
    end
  end
end
