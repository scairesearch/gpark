#!/usr/bin/env bash
# Compile every corpus golden with ptxas and report what it cost.
#
# This is the first gate that knows anything about a real GPU. Everything upstream
# of it — the emitter, the validator, the Python mirror — is checking gpark against
# itself. Only ptxas can tell us the PTX is real.
#
# -v prints per-kernel register usage and, crucially, spill counts. Spills are the
# single most useful number here: they mean a kernel ran out of registers and went
# to local memory, which on a bandwidth-bound kernel destroys the point of the
# kernel. gpark writes register ids by hand, so a too-large kernel is a design
# error, not something the toolchain papers over.
#
# Usage:
#   ptxas_check.sh sm_80              # default corpus
#   ptxas_check.sh sm_90 -j 8         # parallel, 8 jobs
#   ptxas_check.sh sm_80 kernels/extra.ptx
#
# Needs: ptxas on PATH (it ships with the CUDA toolkit).

set -euo pipefail

ARCH="${1:-sm_80}"
shift || true

JOBS="${GPARK_JOBS:-4}"
if [[ "${1:-}" == "-j" ]]; then
  JOBS="$2"
  shift 2
fi

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CORPUS="${HERE}/../corpus/golden"

if [[ $# -gt 0 ]]; then
  FILES=("$@")
else
  FILES=("${CORPUS}"/*.ptx)
fi

# Resolve GPARK_PTXAS *before* the presence check. Checking `command -v ptxas`
# first made the override useless: the error message told you to set
# GPARK_PTXAS, setting it still failed, because the guard had already decided
# ptxas was missing. That is the normal state on a hosted runner and on any
# machine where CUDA lives outside PATH, so it was the common case broken.
PTXAS="${GPARK_PTXAS:-ptxas}"

if [[ ! -x "${PTXAS}" ]] && ! command -v "${PTXAS}" >/dev/null 2>&1; then
  cat >&2 <<EOF
error: no usable ptxas.

Looked for "${PTXAS}".

Install the CUDA toolkit, or point GPARK_PTXAS at the binary. CUDA usually
lands outside PATH, so the explicit path is usually what you want:

    GPARK_PTXAS=/usr/local/cuda/bin/ptxas remote/ptxas_check.sh sm_80

See docs/VALIDATION.md for what this gate does and does not prove.
EOF
  exit 127
fi

if [[ ! -f "${CORPUS}/.gitignore" ]]; then
  mkdir -p "${CORPUS}"
fi
OUT="${CORPUS}/.ptxas"
mkdir -p "${OUT}"

echo "arch:  ${ARCH}"
echo "ptxas: $("${PTXAS}" --version | tail -1)"
echo

# -O3 because that is the level a real deployment uses; a kernel that only
# assembles at -O0 has not been tested.
# --verbose gives register and spill counts, which is the point of the script.
run_one() {
  local ptx="$1"
  local name
  name="$(basename "${ptx}" .ptx)"

  if ! "${PTXAS}" -arch="${ARCH}" -O3 --verbose -lineinfo \
       "${ptx}" -o "${OUT}/${name}.cubin" 2> "${OUT}/${name}.log"; then
    printf '  %-22s FAILED\n' "${name}"
    sed 's/^/      /' "${OUT}/${name}.log"
    return 1
  fi

  local regs spills
  regs="$(grep -oE 'Used [0-9]+ registers' "${OUT}/${name}.log" | grep -oE '[0-9]+' || echo '?')"
  spills="$(grep -oE '[0-9]+ bytes spill stores' "${OUT}/${name}.log" | grep -oE '[0-9]+' || echo 0)"

  printf '  %-22s ok   registers=%-4s spill_stores=%s\n' "${name}" "${regs}" "${spills}"

  # Spills are a hard failure for this project. gpark assigns registers by hand,
  # so a spill means the kernel as written cannot hold its working set.
  if [[ "${spills}" != "0" && "${spills}" != "?" ]]; then
    printf '  %-22s SPILLED: %s bytes went to local memory\n' "${name}" "${spills}"
    return 1
  fi
}
export -f run_one
export ARCH PTXAS OUT

failures=0
if command -v xargs >/dev/null 2>&1 && [[ "${JOBS}" -gt 1 ]]; then
  # shellcheck disable=SC2016
  printf '%s\0' "${FILES[@]}" \
    | xargs -0 -P "${JOBS}" -I{} bash -c 'run_one "$1" || exit 1' _ {} || failures=$((failures + 1))
else
  for f in "${FILES[@]}"; do
    run_one "${f}" || failures=$((failures + 1))
  done
fi

echo
if [[ "${failures}" -gt 0 ]]; then
  echo "${failures} kernel(s) failed. See ${OUT}/*.log"
  exit 1
fi

echo "all ${#FILES[@]} kernel(s) assembled for ${ARCH}"
echo
echo "Next: remote/exec_harness, which JITs these through the driver API and"
echo "compares against CPU references. Assembling is not the same as being right."
