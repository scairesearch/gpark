defmodule Gpark.PTXTest do
  use ExUnit.Case, async: true

  alias Gpark.IR
  alias Gpark.PTX
  alias Gpark.Validate
  alias Gpark.IR.JSON

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

  describe "the corpus" do
    test "every spec round-trips and reproduces its golden byte for byte" do
      # This is the contract the Python implementation has to satisfy: not
      # "close enough", the same bytes. A backend that formats differently is a
      # backend you cannot diff against a real nvcc build.
      for path <- specs() do
        name = Path.basename(path, ".json")
        kernel = path |> File.read!() |> JSON.decode!()

        # decode!/1 returns the kernel itself; only check/1 returns a tuple.
        assert ^kernel = JSON.decode!(JSON.encode!(kernel)),
               "#{name}: spec is not a fixed point of encode/decode"

        assert {:ok, _} = Validate.check(kernel), "#{name}: does not validate"

        golden = File.read!(Path.join(Gpark.Golden.dir(), "#{name}.ptx"))
        assert PTX.emit(kernel) == golden, "#{name}: PTX differs from golden"
      end
    end

    test "every golden has a spec and every spec has a golden" do
      names = Enum.map(specs(), &Path.basename(&1, ".json")) |> Enum.sort()
      goldens = Gpark.Golden.contents() |> Enum.map(&Path.basename(&1, ".ptx")) |> Enum.sort()
      assert names == goldens
    end

    test "bounds guards precede every global access" do
      # A kernel that guards a bounds check by branching *after* the guarded work is
      # silently, catastrophically wrong: every out-of-range lane performs the loads
      # and stores it was supposed to skip. `unpack_u4_f32` had exactly this bug --
      # the branch was the block terminator, so it emitted after the body, and an
      # out-of-range lane wrote 32 bytes past the end of the output buffer.
      #
      # Nothing about that is catchable from the bytes alone: the golden is
      # self-consistent and the validator is happy, because both were handed a
      # program that is valid PTX and merely not the program intended. So the
      # property is asserted directly on the emitted text.
      for name <- kernel_names() do
        ptx = File.read!(Path.join(Gpark.Golden.dir(), "#{name}.ptx"))

        # Match on the index list rather than `assert`ing it: `assert []` passes in
        # Elixir, because only nil and false are falsy.
        branch_at =
          case Regex.run(~r/bra \$L__\w+;/, ptx, return: :index) do
            [{at, _} | _] -> at
            nil -> flunk("#{name}: no bounds guard branch in the emitted PTX")
          end

        # The property that matters for memory safety, and the one every bounds-checked
        # kernel must satisfy: no global *store* may precede the guard. `unpack_u4_f32`
        # had exactly this bug -- the branch was the block terminator, so it emitted
        # after the body, and an out-of-range lane wrote 32 bytes past the output buffer.
        store_at =
          case Regex.run(~r/st\.global/, ptx, return: :index) do
            [{at, _} | _] -> at
            nil -> nil
          end

        if store_at do
          assert store_at > branch_at,
                 """
                 #{name}: first global store at byte #{store_at} precedes the bounds \
                 guard at byte #{branch_at}, so out-of-range lanes would write \
                 out of bounds. Move the guarded branch into the instruction list, \
                 before the body, and make the block terminator `ret`.
                 """
        end

        # Stronger property, for kernels that guard *per element* and therefore must
        # not even read out of range. `reduce_sum_f32` is deliberately excluded: it is
        # a reduction, so it has to load every lane, and it relies on the caller
        # passing a warp-multiple `n`. That contract is documented in the kernel and
        # enforced by the harness padding, but it is a weaker guarantee than an
        # in-kernel guard, and it is the reason this loop is not simply
        # `for name <- kernel_names()`.
        if name in ["vec_add_f32", "saxpy_f32", "unpack_u4_f32"] do
          load_at =
            case Regex.run(~r/ld\.global/, ptx, return: :index) do
              [{at, _} | _] -> at
              nil -> flunk("#{name}: expected a global load in the emitted PTX")
            end

          assert load_at > branch_at,
                 """
                 #{name}: first global load at byte #{load_at} precedes the bounds \
                 guard at byte #{branch_at}, so out-of-range lanes would read \
                 out of bounds.
                 """
        end
      end
    end

    test "covers the intended kernel families" do
      # Guards against a corpus that quietly stops testing anything.
      assert kernel_names() ==
               ["reduce_sum_f32", "saxpy_f32", "unpack_u4_f32", "vec_add_f32"]

      # Each of these is something a naive emitter gets wrong, so each is pinned
      # by opcode shape rather than left to the byte-for-byte comparison alone:
      # 64-bit widening, a fused arithmetic op, warp shuffles, and the two-type
      # conversion that sub-byte work is entirely made of.
      assert File.read!(Path.join(Gpark.Golden.dir(), "vec_add_f32.ptx")) =~ "mul.wide.u32"
      assert File.read!(Path.join(Gpark.Golden.dir(), "saxpy_f32.ptx")) =~ "fma.rn.f32"

      assert File.read!(Path.join(Gpark.Golden.dir(), "reduce_sum_f32.ptx")) =~
               "shfl.sync.bfly.f32"

      quant = File.read!(Path.join(Gpark.Golden.dir(), "unpack_u4_f32.ptx"))
      # A rounding mode *and* a source type, which is what PTX requires.
      assert quant =~ "cvt.rn.f32.u32"
      # Masking a u32 word: only expressible because the bit-container types are
      # in the table's operand set for the bitwise ops.
      assert quant =~ "and.u32"
    end

    defp kernel_names, do: specs() |> Enum.map(&Path.basename(&1, ".json"))

    defp specs do
      Path.join(Gpark.Golden.corpus_dir(), "specs/*.json")
      |> Path.wildcard()
      |> Enum.sort()
    end
  end
end
