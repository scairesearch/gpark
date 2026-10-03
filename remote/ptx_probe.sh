#!/usr/bin/env bash
# Assemble a battery of minimal PTX snippets and report which ones ptxas accepts.
#
# This exists because "Arguments mismatch for instruction 'ld'" from a real
# ptxas is not a bisectable signal. The goldens cascade: one bad construct
# poisons every instruction after it, so the reported error names whichever
# instruction happened to follow the real one. Gpark.Validate cannot help
# because it checks the IR, not PTX. Guessing costs a CI round trip per
# hypothesis; this answers a dozen at once.
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

# Full control over the case text, so a case can vary the header as well as the
# body. $1 = case name, $2 = complete module text.
run_full() {
  local name="$1" text="$2"
  printf '%s\n' "${text}" > "${WORK}/${name}.ptx"
  local out
  if out="$("${PTXAS}" -arch=sm_80 -O3 "${WORK}/${name}.ptx" -o "${WORK}/${name}.cubin" 2>&1)"; then
    printf 'PASS  %s\n' "${name}"
  else
    # Keep the file, line and message: the line number is what makes the
    # cascade readable, and dropping it is what made this unreadable before.
    local msg
    msg="$(printf '%s' "${out}" | grep -m1 -E 'error|fatal' | sed "s|${WORK}/${name}.ptx|<case>|")"
    printf 'FAIL  %-24s %s\n' "${name}" "${msg:0:120}"
  fi
}

# The fixed skeleton: three params matching what the real kernels load.
# $1 = .reg lines, $2 = body.
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

# --- does a module with no .reg at all assemble? --------------------------
# Baseline. If this fails, the problem is in .visible/.param/.target, which
# every case below shares, and no body comparison means anything.
run_full bare_ret '.version 8.7
.target sm_80
.address_size 64
.visible .entry k(.param .u32 n)
{
$L__entry:
	ret;
}'

# --- .reg declaration style ----------------------------------------------
# The real suspect. gpark emits `.reg .u32 %r2<2>;` -- a trailing digit on the
# base name. The documented idiom is `.reg .u32 %r<2>;`, which names %r, %r1.
# If the digit form is what ptxas rejects, every case in this repo fails for
# that reason alone, in every kernel, at the first register write.
run_case reg_nodigit_mov   '	.reg .u32 %r<2>;'    '	mov.u32 %r1, %ctaid.x;'
run_case reg_digit_mov     '	.reg .u32 %r1<2>;'   '	mov.u32 %r1, %ctaid.x;'
run_case reg_digit_off_mov '	.reg .u32 %r2<2>;'   '	mov.u32 %r2, %ctaid.x;'
run_case reg_b32_mov       '	.reg .b32 %r<2>;'    '	mov.u32 %r1, %ctaid.x;'
run_case reg_u32_imm       '	.reg .u32 %r<2>;'    '	mov.u32 %r1, 0;'

# --- 64-bit and float banks ------------------------------------------------
run_case reg_u64_nodigit   '	.reg .u64 %rd<2>;'   '	mov.u64 %rd, %rd;'
run_case reg_u64_digit     '	.reg .u64 %rd1<2>;'  '	mov.u64 %rd1, %rd1;'
run_case reg_f32_nodigit   '	.reg .f32 %f<2>;'    '	mov.f32 %f, %f1;'
run_case reg_f32_digit     '	.reg .f32 %f1<2>;'   '	mov.f32 %f1, %f2;'
run_case reg_pred_nodigit  '	.reg .pred %p<2>;'   '	setp.ge.u32 %p, %r1, %r1;'
run_case reg_pred_digit    '	.reg .pred %p1<2>;'  '	setp.ge.u32 %p1, %r1, %r1;'

# --- the four real shapes, once the reg style is settled ------------------
run_case ld_param_u64 '	.reg .u64 %rd<2>;' '	ld.param.u64 %rd, [x];'
run_case ld_global_f32 '	.reg .u64 %rd<2>;
	.reg .f32 %f<2>;' '	ld.global.f32 %f, [%rd];'
run_case mulwide_u32 '	.reg .u32 %r<2>;
	.reg .u64 %rd<2>;' '	mul.wide.u32 %rd, %r1, 4;'
run_case bra_not '	.reg .pred %p<2>;
	.reg .u32 %r<2>;' '	setp.ge.u32 %p1, %r1, %r1;
	not.pred %p, %p1;
	@%p bra $L__exit;
$L__exit:'