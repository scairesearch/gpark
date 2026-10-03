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

# --- one .reg per register, no vector form -------------------------------
# The shape gpark should emit. gpark allocates registers explicitly and never
# reuses them, so declaring them one per line is both valid and the honest
# encoding of that. This battery asks whether it assembles at all, for every
# type gpark uses, and whether each operand name is accepted.
run_case single_u32      '	.reg .u32 %r1;'  '	mov.u32 %r1, %ctaid.x;'
run_case single_u64      '	.reg .u64 %rd1;' '	mov.u64 %rd1, %rd1;'
run_case single_f32      '	.reg .f32 %f1;
	.reg .f32 %f2;'        '	mov.f32 %f1, %f2;'
run_case single_pred     '	.reg .pred %p1;' '	setp.ge.s32 %p1, %r1, %r1;'
run_case single_pred_u32 '	.reg .pred %p1;
	.reg .u32 %r1;'        '	setp.ge.u32 %p1, %r1, %r1;'

# --- bare vs suffixed operand names, one declaration at a time -------------
# The earlier battery declared '%r<2>' but wrote '%r' in one case and '%r1' in
# another, so "digit declaration form" and "bare operand name" were conflated.
# Separated here: one declaration, one operand name.
run_case nodigit_decl_bare_operand   '	.reg .u32 %r<2>;'  '	mov.u32 %r, 0;'
run_case nodigit_decl_suffix_operand '	.reg .u32 %r<2>;'  '	mov.u32 %r1, 0;'
run_case nodigit_u64_bare            '	.reg .u64 %rd<2>;' '	mov.u64 %rd, %rd;'
run_case nodigit_u64_suffix          '	.reg .u64 %rd<2>;' '	mov.u64 %rd1, %rd1;'
run_case nodigit_f32_bare            '	.reg .f32 %f<2>;'  '	mov.f32 %f, %f1;'
run_case nodigit_f32_suffix          '	.reg .f32 %f<2>;'  '	mov.f32 %f1, %f2;'

# --- the real kernel shapes, declared one per register ---------------------
# If these pass, fixing the emitter is a one-line-per-register change and the
# goldens can be regenerated and gated for real.
run_case shape_ld_param_u64 '	.reg .u64 %rd1;' '	ld.param.u64 %rd1, [x];'
run_case shape_ld_global_f32 '	.reg .u64 %rd1;
	.reg .f32 %f1;'          '	ld.global.f32 %f1, [%rd1];'
run_case shape_mulwide_u32 '	.reg .u32 %r1;
	.reg .u64 %rd1;'        '	mul.wide.u32 %rd1, %r1, 4;'
run_case shape_addr_sum '	.reg .u64 %rd1;
	.reg .u64 %rd2;
	.reg .f32 %f1;'          '	ld.global.f32 %f1, [%rd1+%rd2];'
run_case shape_guard '	.reg .pred %p1;
	.reg .pred %p2;
	.reg .u32 %r1;
	.reg .u32 %r2;'          '	mov.u32 %r1, %ctaid.x;
	setp.ge.u32 %p1, %r1, %r2;
	not.pred %p2, %p1;
	@%p2 bra $L__exit;
$L__exit:'
run_case shape_ld_param_f32 '	.reg .f32 %f1;' '	ld.param.f32 %f1, [alpha];'
run_case shape_ld_param_u32 '	.reg .u32 %r1;' '	ld.param.u32 %r1, [n];'
run_case shape_st_global_f32 '	.reg .u64 %rd1;
	.reg .f32 %f1;'          '	st.global.f32 [%rd1], %f1;'
run_case shape_ret_only '	.reg .u32 %r1;' '	mov.u32 %r1, 0;'