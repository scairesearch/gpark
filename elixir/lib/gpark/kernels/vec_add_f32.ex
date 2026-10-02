defmodule Gpark.Kernels.VecAddF32 do
  @moduledoc """
  `out[i] = a[i] + b[i]` — the reference kernel in the corpus.

  Written the hard way on purpose. gpark emits PTX directly, so there is no
  compiler underneath to rescue a sloppy index calculation: the 64-bit address
  arithmetic and the bounds check are both explicit, because both are the
  difference between a kernel that works on 2^32 elements and one that silently
  wraps at 2^30.

  One thread owns one element, so there is no `bar.sync` and no shared memory
  here. This kernel exists to pin down the end-to-end scaffolding; `vec_add_f32_v4`
  is where vectorisation gets exercised.
  """

  alias Gpark.IR

  # Register budget, assigned by hand. gpark does not re-allocate these; on a
  # register-starved kernel, every spare vector is another load in flight.
  # a pointer
  @rd_a 1
  # b pointer
  @rd_b 2
  # out pointer
  @rd_out 3
  # byte offset
  @rd_off 4
  # n
  @r_n 1
  # global thread id == element index
  @r_gid 2
  # gid >= n
  @p_in 1
  # !p_in
  @p_done 2
  @f_a 1
  @f_b 2
  @f_sum 3

  @doc "Build the `vec_add_f32` kernel IR."
  def build do
    IR.kernel(:vec_add_f32,
      target: "sm_80",
      params: [
        IR.param_decl(:a, :u64),
        IR.param_decl(:b, :u64),
        IR.param_decl(:out, :u64),
        IR.param_decl(:n, :u32)
      ],
      blocks: [entry(), done()]
    )
  end

  defp entry do
    IR.block(
      :entry,
      [
        # Parameters come from the driver's parameter bank, not from constants.
        IR.instr("ld",
          dtype: :u64,
          space: :param,
          dest: IR.reg(:u64, @rd_a),
          ops: [IR.param(:a)]
        ),
        IR.instr("ld",
          dtype: :u64,
          space: :param,
          dest: IR.reg(:u64, @rd_b),
          ops: [IR.param(:b)]
        ),
        IR.instr("ld",
          dtype: :u64,
          space: :param,
          dest: IR.reg(:u64, @rd_out),
          ops: [IR.param(:out)]
        ),
        IR.instr("ld", dtype: :u32, space: :param, dest: IR.reg(:u32, @r_n), ops: [IR.param(:n)]),

        # One element per thread: the block index *is* the element index.
        IR.instr("mov", dtype: :u32, dest: IR.reg(:u32, @r_gid), ops: [IR.sreg(:ctaid_x)]),

        # Bounds check as an explicit early exit rather than a predicated
        # load/store pair. Predication would leave `fa`/`fb` unwritten for
        # out-of-range threads, and the whole point of writing this by hand is
        # that every read has an obvious write.
        IR.instr("setp",
          dtype: :u32,
          modifier: "ge",
          dest: IR.pred(@p_in),
          ops: [IR.reg(:u32, @r_gid), IR.reg(:u32, @r_n)]
        ),
        IR.instr("not", dtype: :pred, dest: IR.pred(@p_done), ops: [IR.pred(@p_in)]),

        # PTX has no "branch if false", so branch on the negated predicate.
        IR.instr("bra", ops: [IR.label(:done)], pred: IR.pred(@p_done)),

        # Byte offset = gid * sizeof(f32). `mul.wide` widens a 32-bit multiply to
        # 64 bits, so `gid * 4` stays correct past 2^30 elements instead of
        # wrapping the way a plain `.u32` multiply would.
        IR.instr("mul.wide",
          dtype: :u32,
          dest: IR.reg(:u64, @rd_off),
          ops: [IR.reg(:u32, @r_gid), IR.imm(4)]
        ),
        IR.instr("ld",
          dtype: :f32,
          space: :global,
          dest: IR.reg(:f32, @f_a),
          ops: [IR.addr(IR.reg(:u64, @rd_a), IR.reg(:u64, @rd_off))]
        ),
        IR.instr("ld",
          dtype: :f32,
          space: :global,
          dest: IR.reg(:f32, @f_b),
          ops: [IR.addr(IR.reg(:u64, @rd_b), IR.reg(:u64, @rd_off))]
        ),
        IR.instr("add",
          dtype: :f32,
          dest: IR.reg(:f32, @f_sum),
          ops: [IR.reg(:f32, @f_a), IR.reg(:f32, @f_b)]
        ),
        IR.instr("st",
          dtype: :f32,
          space: :global,
          ops: [IR.addr(IR.reg(:u64, @rd_out), IR.reg(:u64, @rd_off)), IR.reg(:f32, @f_sum)]
        )
      ],
      IR.instr("ret")
    )
  end

  defp done do
    IR.block(:done, [], IR.instr("ret"))
  end
end
