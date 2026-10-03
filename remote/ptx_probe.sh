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

# --- addressing forms ------------------------------------------------------
# gpark's whole model is base + offset, and `mul.wide.u32` into a .u64 then
# used as an address operand fails to parse. PTX register+register addressing
# is 32-bit-offset specific; this asks which forms actually parse.
run_case addr_single '	.reg .u64 %rd1;
	.reg .f32 %f1;'          '	ld.global.f32 %f1, [%rd1];'
run_case addr_rd_plus_rd64 '	.reg .u64 %rd1;
	.reg .u64 %rd2;
	.reg .f32 %f1;'          '	ld.global.f32 %f1, [%rd1+%rd2];'
run_case addr_rd_plus_r32 '	.reg .u64 %rd1;
	.reg .u32 %r1;
	.reg .f32 %f1;'          '	ld.global.f32 %f1, [%rd1+%r1];'
run_case addr_rd_plus_imm '	.reg .u64 %rd1;
	.reg .f32 %f1;'          '	ld.global.f32 %f1, [%rd1+4];'
run_case addr_r32_plus_rd '	.reg .u64 %rd1;
	.reg .u32 %r1;
	.reg .f32 %f1;'          '	ld.global.f32 %f1, [%r1+%rd1];'

# --- producing the offset --------------------------------------------------
# If register+register addressing needs a 32-bit offset, the emitter has to
# stop widening to .u64. Check what is available for that.
run_case widen_mulwide_u64 '	.reg .u32 %r1;
	.reg .u64 %rd1;'        '	mul.wide.u32 %rd1, %r1, 4;'
run_case offset_u32_mul '	.reg .u32 %r1;
	.reg .u32 %r2;'         '	mul.lo.u32 %r2, %r1, 4;'
run_case widen_cvt '	.reg .u32 %r1;
	.reg .u64 %rd1;'        '	cvt.u64.u32 %rd1, %r1;'
run_case shl_u32 '	.reg .u32 %r1;'            '	shl.b32 %r1, %r1, 2;'

# --- the full saxpy addressing chain, both candidate fixes -----------------
run_case saxpy_offset_u32 '	.reg .u64 %rd1;
	.reg .u32 %r1;
	.reg .u32 %r2;
	.reg .f32 %f1;'          '	mov.u32 %r1, %ctaid.x;
	mul.lo.u32 %r2, %r1, 4;
	ld.global.f32 %f1, [%rd1+%r2];'
run_case saxpy_offset_wide '	.reg .u64 %rd1;
	.reg .u32 %r1;
	.reg .u64 %rd2;
	.reg .f32 %f1;'          '	mov.u32 %r1, %ctaid.x;
	mul.wide.u32 %rd2, %r1, 4;
	ld.global.f32 %f1, [%rd1+%rd2];'