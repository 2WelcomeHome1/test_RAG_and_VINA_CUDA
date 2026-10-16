#!/usr/bin/env bash
# Informational scenario: runs the whole program RUNS times per example with a fixed seed and
# reports, separately for each build, whether repeated runs produce identical output files.
#
# Not a pass/fail check of the fix:
#   - natural run-to-run variation is not required to appear;
#   - the fix changes R_g, so base and fixed outputs are not expected to equal each other;
#     compare the stability within each build.
# Each ligand runs in its own process (fixed order, nothing left over from another ligand),
# so the "large ligand, then smaller ligand in one process" case (--ligand_directory)
# is NOT covered here.
#
# Usage:
#   BASE_BIN=/path/base/Vina-GPU-2-1-CUDA FIXED_BIN=/path/fixed/Vina-GPU-2-1-CUDA ./repro_app.sh
# Either binary may be omitted. Optional env: RUNS (5), SEED (42), CASES ("1u1b 2bm2 2er7"), OUT_DIR.
# Exit codes: 0 all runs completed (read the report), 2 ERROR, 77 SKIP (no binaries or no GPU).

set -u -o pipefail

readonly EXIT_OK=0 EXIT_ERROR=2 EXIT_SKIP=77
readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly V11_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
RUNS="${RUNS:-5}"
SEED="${SEED:-42}"
CASES="${CASES:-1u1b 2bm2 2er7}"
OUT_DIR="${OUT_DIR:-$SCRIPT_DIR/out}"

die() { echo "ERROR: $*" >&2; exit "$EXIT_ERROR"; }

[[ "$RUNS" =~ ^[0-9]+$ ]] && [ "$RUNS" -ge 2 ] || die "RUNS must be an integer >= 2"

VERSIONS=()
for v in BASE FIXED; do
	var="${v}_BIN"
	bin="${!var:-}"
	[ -z "$bin" ] && continue
	[ -x "$bin" ] || die "$var=$bin is not an executable file"
	VERSIONS+=("$(tr 'A-Z' 'a-z' <<< "$v")=$(cd "$(dirname "$bin")" && pwd)/$(basename "$bin")")
done
if [ "${#VERSIONS[@]}" -eq 0 ]; then
	echo "SKIP: set BASE_BIN and/or FIXED_BIN to built Vina-GPU-2-1-CUDA binaries"
	exit "$EXIT_SKIP"
fi
if ! nvidia-smi -L > /dev/null 2>&1; then
	echo "SKIP: no NVIDIA GPU visible (nvidia-smi -L failed)"
	exit "$EXIT_SKIP"
fi

if command -v sha256sum > /dev/null 2>&1; then digest() { sha256sum "$1" | cut -d' ' -f1; }
else digest() { shasum -a 256 "$1" | cut -d' ' -f1; }; fi

# Largest |x|,|y|,|z| difference between the ATOM/HETATM records of two PDBQT files.
max_coord_diff() {
	awk 'NR == FNR { if ($0 ~ /^(ATOM|HETATM)/) a[++na] = substr($0, 31, 24); next }
	     /^(ATOM|HETATM)/ { ++nb; if (nb in a) for (k = 0; k < 3; k++) {
	         d = substr(a[nb], 1 + 8 * k, 8) - substr($0, 31 + 8 * k, 8); if (d < 0) d = -d; if (d > m) m = d } }
	     END { if (na != nb) print "n/a(atoms:" na "/" nb ")"; else printf "%.3f\n", m + 0 }' "$1" "$2"
}

# README: at least 8 MB of stack is needed.
stack="$(ulimit -s)"
if [ "$stack" != unlimited ] && [ "$stack" -lt 8192 ]; then ulimit -s 8192 || die "cannot raise stack limit to 8 MB"; fi

echo "gpu:   $(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader | head -1)"
echo "seed:  $SEED   runs: $RUNS   cases: $CASES"
echo "out:   $OUT_DIR"

REPORT=()
for entry in "${VERSIONS[@]}"; do
	ver="${entry%%=*}"
	bin="${entry#*=}"
	for c in $CASES; do
		dir="$OUT_DIR/$ver/$c"
		rm -rf "$dir" && mkdir -p "$dir" || die "cannot create $dir"
		energies=()
		for i in $(seq 1 "$RUNS"); do
			out="$dir/run_$i.pdbqt"
			echo "running $ver $c $i/$RUNS"
			(cd "$V11_DIR" && "$bin" --config "input_file_example/${c}_config.txt" --seed "$SEED" --out "$out") \
				> "$dir/run_$i.log" 2>&1 || die "$ver $c run $i failed (see $dir/run_$i.log)"
			grep -q 'REMARK VINA RESULT' "$out" 2> /dev/null || die "$ver $c run $i: no results in $out"
			digest "$out" >> "$dir/digests.txt"
			energies+=("$(awk '/REMARK VINA RESULT/ { print $4; exit }' "$out")")
		done
		distinct="$(sort -u "$dir/digests.txt" | wc -l | tr -d ' ')"
		worst=0.000
		for i in $(seq 2 "$RUNS"); do
			d="$(max_coord_diff "$dir/run_1.pdbqt" "$dir/run_$i.pdbqt")"
			case "$d" in n/a*) worst="$d"; break ;; esac
			worst="$(awk -v a="$worst" -v b="$d" 'BEGIN { print (b > a) ? b : a }')"
		done
		verdict=STABLE
		[ "$distinct" -gt 1 ] && verdict=VARIES
		REPORT+=("$(printf '%-6s %-5s %-7s distinct=%s/%s  max_coord_diff_vs_run1=%s A  mode1_energy=[%s]' \
			"$ver" "$c" "$verdict" "$distinct" "$RUNS" "$worst" "${energies[*]}")")
	done
done

echo; echo "=== report (informational) ==="
printf '%s\n' "${REPORT[@]}"
echo "note: STABLE/VARIES compares runs of the same build only; base vs fixed equality is not expected."
echo "note: one ligand per process - the large->small ligand sequence in one process is not covered."
exit "$EXIT_OK"
