defmodule Gpark.Kernels.ReduceSumF32 do
  @moduledoc """
  Sum one warp's worth of f32 values into lane 0.

  The first kernel that needs to know how the hardware actually works, and the
  first place gpark's warp-level story has to be right rather than merely
  correct.

  Reduction strategy, in order of why each step exists:

    1. **`shfl.sync.bfly`** with a stride that halves each round. The butterfly
       exchange gives every lane the full sum in 5 rounds for 32 lanes, and
       critically it needs no shared memory and no barrier. The naive
       down-shift reduction only gives lane 0 the answer; the butterfly gives it
       to everyone, which is what you want when the result feeds back into
       per-lane work.

    2. **`bar.sync`** only because the grid is capped at one block per call in
       v0.1. A real multi-block reduction needs atomics across blocks, which is
       a separate kernel; pretending otherwise here would hide the harder problem
       behind an easy one.

  Why `volatile` is deliberately absent: `shfl.sync` is already
  warp-synchronous, so its operand does not need a memory fence to stay live.

  ## Numerical note

  Butterfly reduction sums in a different order than a sequential sum, so it will
  not be bit-identical to NumPy's `sum`. That is inherent to any parallel
  reduction, not a gpark defect — `docs/VALIDATION.md` specifies a tolerance for
  reduction goldens rather than pretending exact equality is achievable.
  """

  alias Gpark.IR

  @rd_in 1
  @rd_out 2
  @r_n 1
  @r_lane 2
  # n - 1, the clamp bound
  @r_last 3
  # byte offset
  @r_off 4
  # lane != 0
  @p_out 1
  # lane == 0
  @p_skip 2
  @f_v 1
  @f_t 2

  # Reduction rounds for a 32-lane warp: strides 16, 8, 4, 2, 1.
  @strides [16, 8, 4, 2, 1]

  @doc "Build the `reduce_sum_f32` kernel IR."
  def build do
    IR.kernel(:reduce_sum_f32,
      target: "sm_80",
      params: [
        IR.param_decl(:in, :u64),
        IR.param_decl(:out, :u64),
        IR.param_decl(:n, :u32)
      ],
      blocks: [entry(), write_out()]
    )
  end

  defp entry do
    IR.block(
      :entry,
      [
        IR.instr("ld",
          dtype: :u64,
          space: :param,
          dest: IR.reg(:u64, @rd_in),
          ops: [IR.param(:in)]
        ),
        IR.instr("ld",
          dtype: :u64,
          space: :param,
          dest: IR.reg(:u64, @rd_out),
          ops: [IR.param(:out)]
        ),
        IR.instr("ld", dtype: :u32, space: :param, dest: IR.reg(:u32, @r_n), ops: [IR.param(:n)]),
        IR.instr("mov", dtype: :u32, dest: IR.reg(:u32, @r_lane), ops: [IR.sreg(:laneid)]),

        # Clamp the element index to n-1. When n < 32 the high lanes would
        # otherwise read past the end of the input; clamping costs one
        # instruction and keeps every lane in the shuffle tree participating,
        # which is required for `shfl.sync` correctness. The clamped lanes
        # contribute duplicate values, so the caller must pass n rounded up to a
        # warp multiple, or the sum will be wrong.
        IR.instr("sub",
          dtype: :u32,
          dest: IR.reg(:u32, @r_last),
          ops: [IR.reg(:u32, @r_n), IR.imm(1)]
        ),
        IR.instr("min",
          dtype: :u32,
          dest: IR.reg(:u32, @r_lane),
          ops: [IR.reg(:u32, @r_lane), IR.reg(:u32, @r_last)]
        ),
        IR.instr("shl",
          dtype: :u32,
          dest: IR.reg(:u32, @r_off),
          ops: [IR.reg(:u32, @r_lane), IR.imm(2)]
        ),
        IR.instr("ld",
          dtype: :f32,
          space: :global,
          dest: IR.reg(:f32, @f_v),
          ops: [IR.addr(IR.reg(:u64, @rd_in), IR.reg(:u32, @r_off))]
        )
      ] ++
        rounds() ++
        [
          # Skip the store for every lane but 0.
          IR.instr("setp",
            dtype: :u32,
            modifier: "ne",
            dest: IR.pred(@p_out),
            ops: [IR.reg(:u32, @r_lane), IR.imm(0)]
          ),
          IR.instr("not", dtype: :pred, dest: IR.pred(@p_skip), ops: [IR.pred(@p_out)]),
          IR.instr("bra", ops: [IR.label(:write_out)], pred: IR.pred(@p_skip))
        ],
      IR.instr("ret")
    )
  end

  # Five rounds of butterfly exchange. `shfl.sync.bfly.f32` is the whole
  # instruction: warp-synchronous, no memory, no barrier, and the result lands
  # in every lane rather than just the first.
  defp rounds do
    Enum.flat_map(@strides, fn stride ->
      [
        IR.instr("shfl",
          dtype: :f32,
          modifier: "bfly",
          dest: IR.reg(:f32, @f_t),
          ops: [IR.reg(:f32, @f_v), IR.imm(stride), IR.imm(32)]
        ),
        IR.instr("add",
          dtype: :f32,
          dest: IR.reg(:f32, @f_v),
          ops: [IR.reg(:f32, @f_v), IR.reg(:f32, @f_t)]
        )
      ]
    end)
  end

  defp write_out do
    IR.block(
      :write_out,
      [
        # Index 0 of the output, unconditionally: lane 0 writes, every other
        # lane is past the block's single slot.
        IR.instr("st",
          dtype: :f32,
          space: :global,
          ops: [IR.addr(IR.reg(:u64, @rd_out)), IR.reg(:f32, @f_v)]
        )
      ],
      IR.instr("ret")
    )
  end
end
