defmodule Gpark.PTXTest do
  use ExUnit.Case, async: true

  alias Gpark.IR
  alias Gpark.PTX

  describe "opcode rendering" do
    test "renders dotted parts in PTX order" do
      assert PTX.opcode(%{base: "ld", space: :global, modifier: nil, vec: nil, dtype: :f32}) ==
               "ld.global.f32"

      assert PTX.opcode(%{base: "ld", space: :global, modifier: "nc", vec: 4, dtype: :f32}) ==
               "ld.global.nc.v4.f32"

      assert PTX.opcode(%{base: "setp", space: nil, modifier: "ge", vec: nil, dtype: :u32}) ==
               "setp.ge.u32"

      assert PTX.opcode(%{base: "red", space: :global, modifier: "add", vec: nil, dtype: :f32}) ==
               "red.global.add.f32"
    end

    test "raises on an unknown opcode rather than emitting garbage" do
      assert_raise ArgumentError, fn ->
        PTX.opcode(%{base: "not_an_opcode", space: nil, modifier: nil, vec: nil, dtype: nil})
      end
    end
  end

  describe "operand rendering" do
    test "maps types onto the correct PTX register bank" do
      assert PTX.operand(IR.reg(:u32, 3)) == "%r3"
      assert PTX.operand(IR.reg(:u64, 3)) == "%rd3"
      assert PTX.operand(IR.reg(:f32, 3)) == "%f3"
      assert PTX.operand(IR.reg(:f64, 3)) == "%fd3"
      assert PTX.operand(IR.pred(2)) == "%p2"
    end

    test "addresses parameters with brackets, as ld.param requires" do
      assert PTX.operand(IR.param(:a)) == "[a]"
    end

    test "renders special registers with their PTX spelling" do
      assert PTX.operand(IR.sreg(:ctaid_x)) == "%ctaid.x"
      assert PTX.operand(IR.sreg(:laneid)) == "%laneid"
    end

    test "renders the three address forms" do
      base = IR.reg(:u64, 2)
      idx = IR.reg(:u64, 4)

      assert PTX.operand(IR.addr(base)) == "[%rd2]"
      assert PTX.operand(IR.addr(base, 64)) == "[%rd2+64]"
      assert PTX.operand(IR.addr(base, idx)) == "[%rd2+%rd4]"
      assert PTX.operand(IR.addr(base, idx, 4)) == "[%rd2+%rd4*4]"
    end
  end

  describe "float immediates" do
    test "encodes f32 as 0f plus 8 hex digits" do
      assert PTX.hex_float(1.0, :f32) == "0f3F800000"
      assert PTX.hex_float(0.0, :f32) == "0f00000000"
      assert PTX.hex_float(-1.0, :f32) == "0fBF800000"
      assert PTX.hex_float(2.0, :f32) == "0f40000000"
    end

    test "encodes f64 as 0d plus 16 hex digits" do
      assert PTX.hex_float(1.0, :f64) == "0d3FF0000000000000"
      assert PTX.hex_float(0.5, :f64) == "0d3FE0000000000000"
    end
  end

  describe "module structure" do
    test "declares only the registers the kernel uses" do
      ptx = Gpark.Kernels.VecAddF32.build() |> PTX.emit()

      assert ptx =~ ".target sm_80"
      assert ptx =~ ".address_size 64"
      assert ptx =~ ".visible .entry vec_add_f32("
      assert ptx =~ ".reg .f32 %f1<3>;"
    end

    test "contiguous register ids collapse into PTX runs" do
      ptx = Gpark.Kernels.VecAddF32.build() |> PTX.emit()

      # %rd1..%rd4 is one run, not four separate names.
      assert ptx =~ ".reg .u64 %rd1<4>;"
      refute ptx =~ "%rd1, %rd2"
    end
  end

  describe "the corpus golden" do
    test "matches the checked-in PTX byte for byte" do
      expected = File.read!(Path.expand("../../../corpus/golden/vec_add_f32.ptx", __DIR__))
      assert Gpark.Kernels.VecAddF32.build() |> PTX.emit() == expected
    end
  end
end
