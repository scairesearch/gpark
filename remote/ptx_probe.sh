#!/usr/bin/env bash
# Assemble a battery of minimal PTX snippets and report which ones ptxas accepts.
#
# This exists because ptxas errors are not bisectable. The goldens cascade --
# one bad construct poisons every instruction after it, so the reported error
# names whichever instruction happened to follow the real one. Gpark.Validate
# cannot help, because it checks the IR, not PTX. Guessing costs a CI round
# trip per hypothesis; this answers a dozen at once.
#
# Established so far:
#   bare_ret      PASS  module header (.version/.target/.param/.visible) is fine
#   .reg .u32 %r<2>;   PASS  the no-digit vector form gpark does not use
#   .reg .u32 %r1<2>;  FAIL  the digit vector form gpark emits
#   .reg .u32 %r2<2>;  FAIL  ditto, different base index
#
# Output is one line per case: PASS, or the first error ptxas reports.
#
# Usage: ptx_probe.sh /path/to/ptxas

set -uo pipefail

PTXAS="${1:?usage: ptx_probe.sh /path/to/ptxas}"
if [[ ! -x "${PTXAS}" ]]; then
  echo "error: ${PTXAS} is not executable" >&2
  exit 127
fi

WORK="$(mktemp -d)"
trap 'rm -rf "${WORK}"' EXIT

# $1 = case name, $2 = complete module text. Full control so a case can vary
# the .reg block, the body, or both.
run_full() {
  local name="$1" text="$2"
  printf '%s\n' "${text}" > "${WORK}/${name}.ptx"
  local out
  if out="$("${PTXAS}" -arch=sm_80 -O3 "${WORK}/${name}.ptx" -o "${WORK}/${name}.cubin" 2>&1)"; then
    printf 'PASS  %s\n' "${name}"
  else
    # Keep file, line and message. Dropping the line number is what made the
    # first run of this unreadable.
    local msg
    msg="$(printf '%s' "${out}" | grep -m1 -E 'error|fatal' | sed "s|${WORK}/${name}.ptx|<case>|")"
    printf 'FAIL  %-26s %s\n' "${name}" "${msg:0:110}"
  fi
}

# $1 = name, $2 = .reg lines, $3 = body. Fixed three-param skeleton.
run_case() {
  local name="$1" regs="$2" body="$3"
  run_full "${name}" ".version 8.7
.target sm_80
.address_size 64
.visible .entry k(
	.param .u64 x,
	.param .f32 alpha,
	.param .u32 n
)
{
${regs}
\$L__entry:
${body}
	ret;
}"
}

echo "ptxas: $("${PTXAS}" --version | tail -1)"
echo

# --- materialising the address ---------------------------------------------
# Register+register addressing is not a syntax slip to be repaired, it does
# not exist in PTX for ld/st: [%rd+%rd], [%rd+%r] and [%r+%rd] all fail to
# parse. Only [reg] and [reg+imm] are accepted.
#
# That matters because base + runtime-computed offset is gpark's entire
# addressing model: an index that is not known until launch cannot become an
# immediate. So the question is not which spelling is right, it is whether
# the address can be built in a register at all.
run_case addr_add_s64 '	.reg .u64 %rd1;
	.reg .u64 %rd2;
	.reg .f32 %f1;'          '	add.s64 %rd1, %rd1, %rd2;
	ld.global.f32 %f1, [%rd1];'
run_case addr_add_u64 '	.reg .u64 %rd1;
	.reg .u64 %rd2;
	.reg .f32 %f1;'          '	add.u64 %rd1, %rd1, %rd2;
	ld.global.f32 %f1, [%rd1];'
run_case addr_add_imm '	.reg .u64 %rd1;
	.reg .f32 %f1;'          '	add.s64 %rd1, %rd1, 4;
	ld.global.f32 %f1, [%rd1];'

# --- a complete kernel, the shape the emitter should produce ---------------
# saxpy_f32 rebuilt with the address materialised in a register. If this
# assembles, the fix is mechanical: the emitter stops folding base+offset into
# the addressing mode and emits add.s64 into a scratch register instead.
# Writes to that scratch register are part of gpark's design already -- it
# allocates explicitly and never reuses -- so this costs registers, not
# correctness.
run_full full_saxpy '.version 8.7
.target sm_80
.address_size 64
.visible .entry saxpy_f32(
	.param .u64 x,
	.param .u64 y,
	.param .u64 out,
	.param .f32 alpha,
	.param .u32 n
)
{
	.reg .pred %p1;
	.reg .pred %p2;
	.reg .u32 %r1;
	.reg .u32 %r2;
	.reg .u64 %rd1;
	.reg .u64 %rd2;
	.reg .u64 %rd3;
	.reg .u64 %rd4;
	.reg .u64 %rd5;
	.reg .f32 %f1;
	.reg .f32 %f2;
	.reg .f32 %f3;
$L__entry:
	ld.param.u64 %rd1, [x];
	ld.param.u64 %rd2, [y];
	ld.param.u64 %rd3, [out];
	ld.param.f32 %f1, [alpha];
	ld.param.u32 %r1, [n];
	mov.u32 %r2, %ctaid.x;
	setp.ge.u32 %p1, %r2, %r1;
	not.pred %p2, %p1;
	@%p2 bra $L__done;
	mul.wide.u32 %rd4, %r2, 4;
	add.s64 %rd5, %rd1, %rd4;
	ld.global.f32 %f2, [%rd5];
	add.s64 %rd5, %rd2, %rd4;
	ld.global.f32 %f3, [%rd5];
	mul.f32 %f2, %f2, %f1;
	add.f32 %f2, %f2, %f3;
	add.s64 %rd5, %rd3, %rd4;
	st.global.f32 [%rd5], %f2;
$L__done:
	ret;
}'

# --- the ops the remaining goldens need ------------------------------------
run_case op_fma '	.reg .f32 %f1;
	.reg .f32 %f2;
	.reg .f32 %f3;'          '	fma.rn.f32 %f3, %f1, %f2, %f3;'
run_case op_prmt '	.reg .b32 %r1;
	.reg .b32 %r2;
	.reg .b32 %r3;'          '	prmt.b32 %r3, %r1, %r2, 0x5140;'
run_case op_shr_sync '	.reg .u32 %r1;'            '	shr.u32 %r1, %r1, 31;'
run_case op_red '	.reg .u32 %r1;
	.reg .u32 %r2;
	.reg .f32 %f1;'          '	max.u32 %r2, %r1, 32;
	shfl.sync.bfly.b32 %r2, %r2, 16, 32;
	add.f32 %f1, %f1, %f1;'