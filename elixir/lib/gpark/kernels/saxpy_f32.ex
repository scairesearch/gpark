defmodule Gpark.Kernels.SaxpyF32 do
  @moduledoc """
  `out[i] = alpha * x[i] + y[i]` — band-bound, so the interesting part is bytes moved.

  This is the honest first speedup target rather than a GEMM. cuBLAS does not
  have a saxpy, so there is no vendor library to beat; the comparison is against
  a naive launch and against cuBLAS on a *neighbouring* op. At 12 bytes of
  traffic per element this kernel is purely bandwidth-bound, which makes it a
  clean check that gpark's addressing is correct and coalesced — if gpark cannot
  hit roofline here, nothing it does later will mean anything.

  `alpha` arrives as a parameter rather than an immediate because that is the
  shape the caller actually has.
  """

  alias Gpark.IR

  @rd_x 1
  @rd_y 2
  @rd_out 3
  @r_n 2
  @r_gid 3
  @rd_off 4

  # Materialised address, shared across the three accesses.
  @rd_addr 5
  @p_in 1
  @p_done 2
  @f_alpha 1
  @f_y 2
  @f_t 3
  @f_r 4

  @doc "Build the `saxpy_f32` kernel IR."
  def build do
    IR.kernel(:saxpy_f32,
      target: "sm_80",
      params: [
        IR.param_decl(:x, :u64),
        IR.param_decl(:y, :u64),
        IR.param_decl(:out, :u64),
        IR.param_decl(:alpha, :f32),
        IR.param_decl(:n, :u32)
      ],
      blocks: [entry(), done()]
    )
  end

  defp entry do
    IR.block(
      :entry,
      [
        IR.instr("ld",
          dtype: :u64,
          space: :param,
          dest: IR.reg(:u64, @rd_x),
          ops: [IR.param(:x)]
        ),
        IR.instr("ld",
          dtype: :u64,
          space: :param,
          dest: IR.reg(:u64, @rd_y),
          ops: [IR.param(:y)]
        ),
        IR.instr("ld",
          dtype: :u64,
          space: :param,
          dest: IR.reg(:u64, @rd_out),
          ops: [IR.param(:out)]
        ),
        IR.instr("ld",
          dtype: :f32,
          space: :param,
          dest: IR.reg(:f32, @f_alpha),
          ops: [IR.param(:alpha)]
        ),
        IR.instr("ld", dtype: :u32, space: :param, dest: IR.reg(:u32, @r_n), ops: [IR.param(:n)]),
        IR.instr("mov", dtype: :u32, dest: IR.reg(:u32, @r_gid), ops: [IR.sreg(:ctaid_x)]),
        IR.instr("setp",
          dtype: :u32,
          modifier: "ge",
          dest: IR.pred(@p_in),
          ops: [IR.reg(:u32, @r_gid), IR.reg(:u32, @r_n)]
        ),
        IR.instr("not", dtype: :pred, dest: IR.pred(@p_done), ops: [IR.pred(@p_in)]),
        IR.instr("bra", ops: [IR.label(:done)], pred: IR.pred(@p_done)),
        IR.instr("mul.wide",
          dtype: :u32,
          dest: IR.reg(:u64, @rd_off),
          ops: [IR.reg(:u32, @r_gid), IR.imm(4)]
        ),

        # Two loads of 4 bytes each, one store of 4 bytes: 12 bytes moved per
        # element, and the whole kernel exists to move them.
        IR.instr("add",
          dtype: :u64,
          dest: IR.reg(:u64, @rd_addr),
          ops: [IR.reg(:u64, @rd_x), IR.reg(:u64, @rd_off)]
        ),
        IR.instr("ld",
          dtype: :f32,
          space: :global,
          dest: IR.reg(:f32, @f_y),
          ops: [IR.addr(IR.reg(:u64, @rd_addr))]
        ),
        IR.instr("add",
          dtype: :u64,
          dest: IR.reg(:u64, @rd_addr),
          ops: [IR.reg(:u64, @rd_y), IR.reg(:u64, @rd_off)]
        ),
        IR.instr("ld",
          dtype: :f32,
          space: :global,
          dest: IR.reg(:f32, @f_t),
          ops: [IR.addr(IR.reg(:u64, @rd_addr))]
        ),

        # alpha * x + y, fused. Splitting this into a separate multiply and add
        # would round twice and cost an extra register.
        IR.instr("fma",
          dtype: :f32,
          modifier: "rn",
          dest: IR.reg(:f32, @f_r),
          ops: [IR.reg(:f32, @f_alpha), IR.reg(:f32, @f_y), IR.reg(:f32, @f_t)]
        ),
        IR.instr("add",
          dtype: :u64,
          dest: IR.reg(:u64, @rd_addr),
          ops: [IR.reg(:u64, @rd_out), IR.reg(:u64, @rd_off)]
        ),
        IR.instr("st",
          dtype: :f32,
          space: :global,
          ops: [IR.addr(IR.reg(:u64, @rd_addr)), IR.reg(:f32, @f_r)]
        )
      ],
      IR.instr("ret")
    )
  end

  defp done do
    IR.block(:done, [], IR.instr("ret"))
  end
end
