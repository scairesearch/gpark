defmodule Gpark.Kernels.UnpackU4F32 do
  @moduledoc """
  Unpack a `u4` weight tensor to `f32`: 8 values per `u32` word, one thread per word.

  This is the kernel gpark exists for. Triton has no 4-bit arithmetic, so a
  quantized kernel there means upcasting by hand and hoping the compiler keeps the
  shape; Taichi has no sub-byte primitive at all, only an optional `quant`
  extension that does not lower to hardware. Here the unpack is explicit and the
  cost of that explicitness is visible:

      4 bytes in, 32 bytes out

  which is why this is a *dequantise* kernel and not a matmul. The interesting
  number is the output bytes, and it is fixed — the only lever is issuing the
  loads and stores well enough not to become the bottleneck. Anything slower than
  that means the addressing is wrong, not the algorithm.

  ## How the bit twiddling works

  One `u32` word holds 8 unsigned 4-bit values, low nibble first. For element `k`:

      t = (word >> (4 * k)) & 0xF

  Note the `.b32` suffix on the shift and the mask. That is why `and`/`shl`/`shr`
  accept the bit-container types: masking a `.b32` word is not expressible in the
  signed/unsigned integer types, so without that the unpack simply cannot be
  written. `and.b32` and `shr.b32` are the entire reason `@bit_ops` exists.

  ## Why the loop is unrolled rather than a real loop

  Eight copies of shift/mask/convert/scale/store, unrolled. A loop would need a
  phi node and a dynamic trip count, and gpark v0.1 has no SSA and no phi — by
  design, because the register allocator in `Gpark.Mid` is where that belongs. The
  unrolled form costs registers, which is exactly the tradeoff the allocator will
  later be asked to get right, and the honest thing is to leave the problem visible.

  The `bfi`/`prmt`/`bfe` instructions would cut the instruction count substantially
  on real hardware; they are deferred (`docs/PTX-SUBSET.md`).
  """

  alias Gpark.IR

  @rd_in 1
  @rd_out 2
  @r_n 1
  @r_gid 2
  @rd_word_off 3
  @r_word 4
  @r_shift 5
  @r_tmp 6
  # byte offset of this thread's first output f32
  @rd_out_idx 4

  # Materialised address, shared by the word load and the element store.
  @rd_addr 5
  @p_in 1
  @p_skip 2
  @f_scale 1
  @f_v 2

  @mask 0xF
  @elems_per_word 8
  @in_bytes_per_word 4
  @out_bytes_per_word 32

  @doc "Build the `unpack_u4_f32` kernel IR."
  def build do
    IR.kernel(:unpack_u4_f32,
      target: "sm_80",
      params: [
        IR.param_decl(:in, :u64),
        IR.param_decl(:out, :u64),
        IR.param_decl(:scale, :f32),
        IR.param_decl(:n, :u32)
      ],
      blocks: [entry(), done()]
    )
  end

  defp entry do
    # The guarded branch is an *instruction*, not the block terminator, and it must
    # come before any memory access. PTX has no "branch if false", so branch on the
    # negated predicate; an untaken predicated branch falls through to the next
    # instruction, which is the body. Using the block terminator for the branch
    # instead would emit it *after* the body, so every out-of-range lane would run
    # the work it was supposed to skip -- including a 32-byte out-of-bounds store.
    guard = IR.instr("bra", ops: [IR.label(:done)], pred: IR.pred(@p_skip))

    IR.block(
      :entry,
      bounds_check() ++ [guard] ++ load_word() ++ words() ++ store_index(),
      IR.instr("ret")
    )
  end

  defp bounds_check do
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
      IR.instr("ld",
        dtype: :f32,
        space: :param,
        dest: IR.reg(:f32, @f_scale),
        ops: [IR.param(:scale)]
      ),
      IR.instr("ld", dtype: :u32, space: :param, dest: IR.reg(:u32, @r_n), ops: [IR.param(:n)]),
      IR.instr("mov", dtype: :u32, dest: IR.reg(:u32, @r_gid), ops: [IR.sreg(:ctaid_x)]),
      IR.instr("setp",
        dtype: :u32,
        modifier: "ge",
        dest: IR.pred(@p_in),
        ops: [IR.reg(:u32, @r_gid), IR.reg(:u32, @r_n)]
      ),
      # @p_in means "this thread is out of range" (gid >= n), which is the
      # opposite of what the name suggests at the point of use, so invert it once
      # here rather than at every branch.
      IR.instr("not", dtype: :pred, dest: IR.pred(@p_skip), ops: [IR.pred(@p_in)])
    ]
  end

  defp load_word do
    [
      IR.instr("mul.wide",
        dtype: :u32,
        dest: IR.reg(:u64, @rd_word_off),
        ops: [IR.reg(:u32, @r_gid), IR.imm(@in_bytes_per_word)]
      ),
      IR.instr("add",
        dtype: :u64,
        dest: IR.reg(:u64, @rd_addr),
        ops: [IR.reg(:u64, @rd_in), IR.reg(:u64, @rd_word_off)]
      ),
      IR.instr("ld",
        dtype: :u32,
        space: :global,
        dest: IR.reg(:u32, @r_word),
        ops: [IR.addr(IR.reg(:u64, @rd_addr))]
      ),
      # Output element index for the low nibble. Widened to 64 bits *after* the
      # multiply, because gid * 32 overflows u32 long before the array does.
      IR.instr("mul.wide",
        dtype: :u32,
        dest: IR.reg(:u64, @rd_out_idx),
        ops: [IR.reg(:u32, @r_gid), IR.imm(@out_bytes_per_word)]
      )
    ]
  end

  # Eight unrolled (shift, mask, convert, scale) blocks. `k` is the nibble index.
  defp words do
    Enum.flat_map(0..(@elems_per_word - 1), fn k ->
      shift = IR.instr("mov", dtype: :u32, dest: IR.reg(:u32, @r_shift), ops: [IR.imm(4 * k)])

      # Element 0 is already in place: shifting by zero and masking it would be
      # two instructions that buy nothing, so nibble 0 reads the word directly.
      {extract, source} =
        if k == 0 do
          {[], @r_word}
        else
          {[
             shift,
             IR.instr("shr",
               dtype: :u32,
               dest: IR.reg(:u32, @r_tmp),
               ops: [IR.reg(:u32, @r_word), IR.reg(:u32, @r_shift)]
             ),
             IR.instr("and",
               dtype: :u32,
               dest: IR.reg(:u32, @r_tmp),
               ops: [IR.reg(:u32, @r_tmp), IR.imm(@mask)]
             )
           ], @r_tmp}
        end

      scale =
        IR.instr("mul",
          dtype: :f32,
          dest: IR.reg(:f32, @f_v),
          ops: [IR.reg(:f32, @f_scale), IR.reg(:f32, @f_v)]
        )

      [
        extract,
        [
          IR.instr("cvt",
            dtype: :f32,
            modifier: "rn",
            srctype: :u32,
            dest: IR.reg(:f32, @f_v),
            ops: [IR.reg(:u32, source)]
          ),
          scale,
          IR.instr("add",
            dtype: :u64,
            dest: IR.reg(:u64, @rd_addr),
            ops: [IR.reg(:u64, @rd_out), IR.reg(:u64, @rd_out_idx)]
          ),
          IR.instr("st",
            dtype: :f32,
            space: :global,
            ops: [
              IR.addr(IR.reg(:u64, @rd_addr)),
              IR.reg(:f32, @f_v)
            ]
          )
        ]
      ]
      |> List.flatten()
    end)
  end

  # Advance the output pointer by one f32 for the next nibble. Done with an
  # address-mode offset rather than a new base register so the whole eight-way
  # unroll costs two address registers instead of eight.
  defp store_index do
    [
      IR.instr("add",
        dtype: :u64,
        dest: IR.reg(:u64, @rd_out_idx),
        ops: [IR.reg(:u64, @rd_out_idx), IR.imm(4)]
      )
    ]
  end

  defp done do
    IR.block(:done, [], IR.instr("ret"))
  end
end
