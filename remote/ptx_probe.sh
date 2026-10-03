#!/usr/bin/env bash
# Assemble a battery of minimal PTX snippets and report which ones ptxas accepts.
#
# This exists because "Arguments mismatch for instruction 'ld'" from a real
# ptxas is not a bisectable signal: the corpus goldens cascade, one bad
# construct poisons every instruction after it, and the validator cannot see
# any of it because the IR it checks is not PTX. Guessing from the error text
# costs a CI round trip per hypothesis.
#
# Each case below is the smallest kernel that isolates one question. Output is
# one line per case: PASS, or the first error ptxas reports.
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

# $1 = case name, $2 = body lines. Header is fixed: the three param types the
# kernels in this repo actually load (.u64 pointers, .f32, .u32), plus the
# register banks the bodies refer to.
emit_case() {
  local name="$1" body="$2"
  cat > "${WORK}/${name}.ptx" <<EOF
.version 8.7
.target sm_80
.address_size 64
.visible .entry k(
	.param .u64 x,
	.param .f32 alpha,
	.param .u32 n
)
{
	.reg .pred %p1<2>;
	.reg .u32 %r1<2>;
	.reg .u64 %rd1<2>;
	.reg .f32 %f1<2>;
\$L__entry:
${body}
	ret;
}
EOF
}

run_case() {
  local name="$1" body="$2"
  emit_case "${name}" "${body}"
  local out
  if out="$("${PTXAS}" -arch=sm_80 -O3 "${WORK}/${name}.ptx" -o "${WORK}/${name}.cubin" 2>&1)"; then
    printf 'PASS  %s\n' "${name}"
  else
    printf 'FAIL  %-26s %s\n' "${name}" "$(printf '%s' "${out}" | grep -m1 -oE '(error|fatal)[^$]*' | cut -c1-90)"
  fi
}

echo "ptxas: $("${PTXAS}" --version | tail -1)"
echo

# --- does the header alone assemble? -------------------------------------
run_case header_only '	mov.u32 %r1, %ctaid.x;'

# --- the reported failure: ld.param ---------------------------------------
run_case ld_param_u64 '	ld.param.u64 %rd1, [x];'
run_case ld_param_u32 '	ld.param.u32 %r1, [n];'
run_case ld_param_f32 '	ld.param.f32 %f1, [alpha];'
run_case ld_param_s64 '	ld.param.s64 %rd1, [x];'

# --- is it the .param space qualifier, or the .u64 type? ------------------
run_case ld_u64_nospace '	ld.u64 %rd1, [x];'
run_case ld_global_u64 '	ld.global.u64 %rd1, [%rd1];'
run_case ld_global_u32 '	ld.global.u32 %r1, [%rd1];'
run_case ld_global_f32 '	ld.global.f32 %f1, [%rd1];'

# --- is it the register class? --------------------------------------------
run_case mov_u64 '	mov.u64 %rd1, %rd1;'
run_case mov_u32 '	mov.u32 %r1, %r1;'

# --- is it mul.wide specifically? -----------------------------------------
run_case mulwide_u32 '	mul.wide.u32 %rd1, %r1, 4;'
run_case mulwide_s32 '	mul.wide.s32 %rd1, %r1, 4;'

# --- predication ----------------------------------------------------------
run_case setp_not_bra '	setp.ge.u32 %p1, %r1, %r1;
	not.pred %p2, %p1;
	@%p2 bra $L__exit;
$L__exit:'

# --- addressing forms -----------------------------------------------------
run_case addr_base_plus '	ld.global.f32 %f1, [%rd1+%rd1];'
run_case addr_base_only '	ld.global.f32 %f1, [%rd1];'