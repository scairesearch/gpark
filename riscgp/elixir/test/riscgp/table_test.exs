defmodule RiscGP.TableTest do
  use ExUnit.Case, async: true

  alias RiscGP.Table

  test "the table loads and is not frozen" do
    assert {"RVGPU", "1.0-draft"} = Table.isa()
    refute Table.frozen?(), "the plan forbids freezing the ISA before the P1 gate"
  end

  test "opcode names are unique" do
    names = Table.names()
    assert length(names) == length(Enum.uniq(names))
  end

  test "every opcode carries provenance and timing" do
    for {name, spec} <- Table.opcodes() do
      assert spec.domain in ~w(core dma mop vec sync), "#{name} has no domain"
      assert spec.path in ~w(A B AB), "#{name} has no path"
      assert is_integer(spec.latency) and spec.latency >= 1, "#{name} has no latency"
      assert is_integer(spec.throughput) and spec.throughput >= 1, "#{name} has no throughput"
      assert is_atom(spec.unit), "#{name} has no unit"
    end
  end

  test "throughput is never below latency" do
    # A unit that accepts a new instruction faster than the previous one
    # completes would be modelling a fully pipelined unit that the datapath
    # does not have. For this ISA that is never true.
    for {name, spec} <- Table.opcodes(), spec.throughput < spec.latency do
      flunk("#{name} throughput #{spec.throughput} < latency #{spec.latency}")
    end
  end

  test "both datapaths are represented" do
    assert Table.by_path(:a) != Table.by_path(:b)
    assert "vfmacc" in Table.by_path(:a)
    assert "mop.mma" in Table.by_path(:b)
  end

  test "the three-thread pattern's scarce unit is the matrix" do
    # mop.mma is the number the whole three-thread pattern is built around.
    assert Table.throughput("mop.mma") == 32
    assert Table.latency("mop.mma") == 16
  end

  test "register files report their sizes" do
    assert Table.reg_count(:dst) == 4
    assert Table.reg_count(:srca) == 2
    assert Table.reg_count("vec") == 32
    assert Table.reg_count(:nope) == 0
  end
end
