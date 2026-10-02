defmodule RiscGP.Kernels do
  @moduledoc """
  Reference RVGPU kernels, in the shape the riscgp plan's three P0 workloads
  need.

  `tiled_mma` is the important one: it is the three-thread unpack / math / pack
  pattern that Tenstorrent's documentation describes as the way to keep the
  single Matrix Unit fed, written out explicitly. It is deliberately *not* a
  neat loop over threads — that is the shape a programmer wants to write, and
  it is exactly the shape that starves the one unit the whole tile is built
  around.
  """

  alias RiscGP.IR

  @doc """
  A `mop.mma` in the middle of the unpack/math/pack pattern.

  `:t0` loads the left operand, `:t1` issues the multiply-accumulate, `:t2`
  writes the result back. Every coprocessor read is separated from its write by
  a `stallwait`, so this kernel passes validation; removing one is the negative
  test that proves the hazard check works.
  """
  def tiled_mma do
    IR.kernel(:tiled_mma,
      path: :b,
      params: [IR.param_decl(:in, :u32), IR.param_decl(:out, :u32)],
      blocks: [
        IR.block(:setup, :t0,
          [
            IR.instr("mop.set", thread: :t0, modifier: "lreg", space: :sram, dtype: :f32,
              ops: [IR.imm(0), IR.addr(IR.reg(:u32, 5), IR.imm(64), 4)]),
            IR.instr("mop.set", thread: :t0, modifier: "srca", space: :sram, dtype: :f32,
              ops: [IR.imm(0), IR.addr(IR.reg(:u32, 5), IR.imm(128), 4)]),
            IR.instr("sync.sem", thread: :t0, modifier: "release", ops: [IR.sem(0), IR.imm(1)])
          ],
          IR.instr("jal", thread: :t0, ops: [IR.label(:setup)])
        ),

        IR.block(:math, :t1,
          [
            IR.instr("sync.sem", thread: :t1, modifier: "acquire", ops: [IR.sem(0), IR.imm(1)]),
            IR.instr("stallwait", thread: :t1, modifier: "unit", ops: [IR.imm(2), IR.imm(0)]),
            IR.instr("mop.mma", thread: :t1, modifier: "acc", dtype: :f32,
              dest: IR.dst(0), ops: [IR.lreg(0), IR.srca(0), IR.srcb(0)]),
            IR.instr("stallwait", thread: :t1, modifier: "unit", ops: [IR.imm(2), IR.dst(0)]),
            IR.instr("mop.store", thread: :t1, space: :sram, dtype: :f32,
              ops: [IR.sem(1), IR.addr(IR.reg(:u32, 6), IR.imm(256), 4)])
          ],
          nil
        ),

        IR.block(:pack, :t2,
          [
            IR.instr("sync.sem", thread: :t2, modifier: "acquire", ops: [IR.sem(1), IR.imm(1)]),
            IR.instr("stallwait", thread: :t2, modifier: "unit", ops: [IR.imm(4), IR.sem(1)]),
            IR.instr("mop.store", thread: :t2, space: :sram, dtype: :f32,
              ops: [IR.sem(1), IR.addr(IR.reg(:u32, 6), IR.imm(256), 4)])
          ],
          nil
        )
      ]
    )
  end

  @doc """
  A tight `mop.mma` loop issued from a single thread.

  This is the shape that looks better and runs worse: one thread does unpack,
  math and pack in turn, so the Matrix Unit idles for the unpack and pack
  phases. Comparing its cycle count against `tiled_mma` is the cheapest possible
  demonstration of why the three-thread pattern exists.
  """
  def serial_mma do
    IR.kernel(:serial_mma,
      path: :b,
      params: [IR.param_decl(:n, :u32)],
      blocks: [
        IR.block(:body, :t1,
          [
            IR.instr("mop.set", thread: :t1, modifier: "srca", space: :sram, dtype: :f32,
              ops: [IR.imm(0), IR.addr(IR.reg(:u32, 5), IR.imm(0), 4)]),
            IR.instr("stallwait", thread: :t1, modifier: "unit", ops: [IR.imm(2), IR.srca(0)]),
            IR.instr("mop.mma", thread: :t1, modifier: "acc", dtype: :f32,
              dest: IR.dst(0), ops: [IR.lreg(0), IR.srca(0), IR.srcb(0)]),
            IR.instr("stallwait", thread: :t1, modifier: "unit", ops: [IR.imm(2), IR.dst(0)]),
            IR.instr("mop.store", thread: :t1, space: :sram, dtype: :f32,
              ops: [IR.sem(0), IR.addr(IR.reg(:u32, 6), IR.imm(0), 4)])
          ],
          IR.instr("jal", thread: :t1, ops: [IR.label(:body)])
        )
      ]
    )
  end
end
