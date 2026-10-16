#!/usr/bin/env bash
# Builds test_gyration.cu twice and checks both builds:
#   base  - inc/cuda/mutate_conf.cuh taken from BASE_REF (the unfixed upstream code);
#   fixed - inc/cuda/mutate_conf.cuh from the working tree.
#
# Usage:  BASE_REF=<commit> ./run_test.sh      or      ./run_test.sh <commit>
#   Verified upstream base: 7b09586a64025d06c854a63eee21268fba5bc8f0 (HPCLab-933/Vina-CUDA, master)
# Optional env: NVCC (default nvcc), CUDA_ARCH (e.g. sm_86), COMPUTE_SANITIZER, BUILD_DIR.
#
# Exit codes: 0 PASS, 1 FAIL, 2 ERROR (build/CUDA/missing results), 77 SKIP (no nvcc or GPU).
# PASS means: base shows exactly the expected defect and fixed matches the reference.

set -u -o pipefail

readonly EXIT_PASS=0 EXIT_FAIL=1 EXIT_ERROR=2 EXIT_SKIP=77
readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly V11_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
readonly SIZES=(16 31 32 55 127)

die() { echo "ERROR: $*" >&2; exit "$EXIT_ERROR"; }

BASE_REF="${1:-${BASE_REF:-}}"
[ -n "$BASE_REF" ] || die "BASE_REF is required (verified base: 7b09586a64025d06c854a63eee21268fba5bc8f0)"

REPO_ROOT="$(git -C "$V11_DIR" rev-parse --show-toplevel 2>/dev/null)" || die "$V11_DIR is not inside a git repository"
HEADER_REL="$(git -C "$V11_DIR" rev-parse --show-prefix)inc/cuda/mutate_conf.cuh" # path relative to the repo root
BASE_COMMIT="$(git -C "$REPO_ROOT" rev-parse --verify --quiet "${BASE_REF}^{commit}")" || die "cannot resolve BASE_REF '$BASE_REF'"

BUILD_DIR="${BUILD_DIR:-$SCRIPT_DIR/build}"
mkdir -p "$BUILD_DIR/base" || die "cannot create $BUILD_DIR"
git -C "$REPO_ROOT" show "$BASE_COMMIT:$HEADER_REL" > "$BUILD_DIR/base/mutate_conf.cuh" \
	|| die "git show $BASE_COMMIT:$HEADER_REL failed"

echo "repo:       $REPO_ROOT"
echo "base:       $BASE_COMMIT:$HEADER_REL"
echo "fixed:      working tree $V11_DIR/inc/cuda/mutate_conf.cuh"
if cmp -s "$BUILD_DIR/base/mutate_conf.cuh" "$V11_DIR/inc/cuda/mutate_conf.cuh"; then
	echo "warning:    working tree header is identical to base (fix not applied)"
fi

NVCC="${NVCC:-nvcc}"
if ! command -v "$NVCC" > /dev/null 2>&1; then
	echo "SKIP: $NVCC not found; nothing was built or run"
	exit "$EXIT_SKIP"
fi
# kernel2.h needs a docking box macro (Makefile: DOCKING_BOX_SIZE); it only sizes grid structs.
NVCC_FLAGS=(-O2 -lineinfo -std=c++14 -DLARGE_BOX)
[ -n "${CUDA_ARCH:-}" ] && NVCC_FLAGS+=(-arch="$CUDA_ARCH")

# Exactly one directory with mutate_conf.cuh is on the include path of each build;
# commonMacros.h and kernel2.h always come from the working tree inc/.
build() { # name header_dir
	"$NVCC" "${NVCC_FLAGS[@]}" -I"$2" -I"$V11_DIR/inc" "$SCRIPT_DIR/test_gyration.cu" \
		-o "$BUILD_DIR/test_gyration_$1" > "$BUILD_DIR/build_$1.log" 2>&1 \
		|| { cat "$BUILD_DIR/build_$1.log" >&2; die "build '$1' failed (see $BUILD_DIR/build_$1.log)"; }
}
build base "$BUILD_DIR/base"
build fixed "$V11_DIR/inc/cuda"

# field LINE KEY -> value of KEY=value in a RESULT line
field() { sed -n "s/.* $2=\([^ ]*\).*/\1/p" <<< "$1"; }

# check_build NAME EXPECT(defect|clean) -> sets STATUS to PASS/FAIL, exits on ERROR/SKIP
check_build() {
	local name="$1" expect="$2" log="$BUILD_DIR/run_$1.log" rc
	"$BUILD_DIR/test_gyration_$name" --mode prefill > "$log" 2>&1
	rc=$?
	cat "$log"
	if [ "$rc" -eq "$EXIT_SKIP" ]; then
		echo "SKIP: no usable CUDA device"
		exit "$EXIT_SKIP"
	fi
	[ "$rc" -eq 0 ] || [ "$rc" -eq 1 ] || die "test_gyration_$name exited with $rc (not a test verdict)"

	local problems=() pattern n line vs_ref vs_orig
	for pattern in zero nonzero; do
		for n in "${SIZES[@]}"; do
			line="$(grep "^RESULT mode=prefill pattern=$pattern n=$n " "$log")"
			[ "$(grep -c . <<< "$line")" -eq 1 ] || die "$name: expected one RESULT line for pattern=$pattern n=$n"
			[ "$(field "$line" finite)" = yes ] || problems+=("$pattern n=$n: non-finite result")
			[ "$(field "$line" threads_agree)" = yes ] || problems+=("$pattern n=$n: threads disagree")
			vs_ref="$(field "$line" vs_ref)"
			vs_orig="$(field "$line" vs_orig_model)"
			if [ "$expect" = clean ] || [ "$pattern" = zero ]; then
				[ "$vs_ref" = MATCH ] || problems+=("$pattern n=$n: expected MATCH with reference, got $vs_ref")
			else
				[ "$vs_ref" = MISMATCH ] || problems+=("$pattern n=$n: expected MISMATCH with reference (defect), got $vs_ref")
				[ "$vs_orig" = MATCH ] || problems+=("$pattern n=$n: result does not match the model of the original code")
			fi
		done
	done
	local expected_rc=0
	[ "$expect" = defect ] && expected_rc=1
	[ "$rc" -eq "$expected_rc" ] || problems+=("exit code $rc, expected $expected_rc")

	if [ "${#problems[@]}" -eq 0 ]; then
		STATUS=PASS
	else
		STATUS=FAIL
		printf '  %s\n' "${problems[@]}"
	fi
}

echo; echo "=== base build: expecting the defect (zero tail MATCH, nonzero tail MISMATCH) ==="
check_build base defect; BASE_STATUS=$STATUS
echo; echo "=== fixed build: expecting MATCH for every case ==="
check_build fixed clean; FIXED_STATUS=$STATUS

# compute-sanitizer initcheck on shared memory, run without prefill.
SAN_BASE=SKIP SAN_FIXED=SKIP SAN_NOTE=""
SANITIZER="${COMPUTE_SANITIZER:-compute-sanitizer}"
if ! command -v "$SANITIZER" > /dev/null 2>&1; then
	SAN_NOTE="$SANITIZER not found"
elif ! "$SANITIZER" --help 2>&1 | grep -A4 -- '--initcheck-address-space' | grep -q shared; then
	SAN_NOTE="this $SANITIZER has no '--initcheck-address-space shared'"
else
	sanitize() { # name -> prints the error count
		local log="$BUILD_DIR/sanitizer_$1.log"
		"$SANITIZER" --tool initcheck --initcheck-address-space shared --print-limit 20 \
			"$BUILD_DIR/test_gyration_$1" --mode raw > "$log" 2>&1
		sed -n 's/.*ERROR SUMMARY: \([0-9][0-9]*\) error.*/\1/p' "$log" | tail -1
	}
	errs_base="$(sanitize base)"
	errs_fixed="$(sanitize fixed)"
	[ -n "$errs_base" ] && [ -n "$errs_fixed" ] || die "compute-sanitizer produced no ERROR SUMMARY (see $BUILD_DIR/sanitizer_*.log)"
	echo; echo "=== compute-sanitizer initcheck (shared), --mode raw ==="
	echo "base:  $errs_base error(s)"; grep -m3 'mutate_conf.cuh' "$BUILD_DIR/sanitizer_base.log"
	echo "fixed: $errs_fixed error(s)"
	if [ "$errs_base" -gt 0 ] && grep -q 'mutate_conf.cuh' "$BUILD_DIR/sanitizer_base.log"; then SAN_BASE=PASS; else SAN_BASE=FAIL; fi
	if [ "$errs_fixed" -eq 0 ]; then SAN_FIXED=PASS; else SAN_FIXED=FAIL; fi
fi

echo; echo "=== summary ==="
echo "[$BASE_STATUS] base:  expect the defect: nonzero tail changes R_g, zero tail does not"
echo "[$FIXED_STATUS] fixed: expect R_g = CPU reference for both tails and all sizes"
echo "[$SAN_BASE] sanitizer base:  expect an uninitialized shared read in mutate_conf.cuh${SAN_NOTE:+ ($SAN_NOTE)}"
echo "[$SAN_FIXED] sanitizer fixed: expect no uninitialized shared reads${SAN_NOTE:+ ($SAN_NOTE)}"
echo "logs: $BUILD_DIR"

for s in "$BASE_STATUS" "$FIXED_STATUS" "$SAN_BASE" "$SAN_FIXED"; do
	[ "$s" = FAIL ] && exit "$EXIT_FAIL"
done
exit "$EXIT_PASS"
