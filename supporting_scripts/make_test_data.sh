#!/usr/bin/env bash
#
# make_test_data.sh
#
# Build a small, self-contained test case in test_data/ by subsetting the
# compiled wide-format CSVs in raw_data/ down to a handful of samples.
#
# The point is that a referee (or you, on a fresh machine) can clone the repo
# and run the whole pipeline end to end in seconds without the full dataset:
#
#     bash supporting_scripts/make_test_data.sh 3
#     python 01_biomechanical_processing.py --raw-dir test_data --results-dir results/test
#
# Sample bases are discovered from the SetName column of each sample, so nothing
# is hardcoded. Every row is kept (the segments must stay intact); only columns
# are subsetted.
#
# NOTE: Excel truncates column names at 31 characters, so the '_SetName' suffix
# arrives clipped by however long the sample name is:
#   "210330 MRC Sample A1Data_SetNam"    -> SetNam
#   "210330 MRC Sample A10Data_SetNa"    -> SetNa
#   "210330 MRC Sample C3.2Data_SetN"    -> SetN
# The patterns below therefore match any truncation of '_SetName' rather than
# the literal suffix.
#
# Requires csvkit for quote-safe column selection:
#     pip install csvkit          # or: conda install -c conda-forge csvkit
#
# Usage: bash supporting_scripts/make_test_data.sh [N_SAMPLES]   (default 3)

set -euo pipefail

RAW="raw_data"
OUT="test_data"
N="${1:-3}"

FORCE="compiled_Force_N_phase_140526.1.csv"
DISP="compiled_Displacement_mm_phase_140526.1.csv"
SIZE="compiled_Size_mm_phase_140526.1.csv"
META="compiled_metadata_with_filenames.csv"

command -v csvcut >/dev/null 2>&1 || {
  echo "ERROR: csvcut not found. Install csvkit:  pip install csvkit" >&2
  exit 1
}

for f in "$FORCE" "$DISP" "$SIZE" "$META"; do
  [[ -f "$RAW/$f" ]] || { echo "ERROR: $RAW/$f not found." >&2; exit 1; }
done

mkdir -p "$OUT"

# ---------------------------------------------------------------------
# 1. Discover the sample base names from the force file header.
# ---------------------------------------------------------------------
# csvcut -n prints "  1: Column name" for each column, which handles quoting and
# embedded commas properly; sed strips the index and the _SetName suffix.
# '_Set', '_SetN', '_SetNa', '_SetNam' and '_SetName' are all accepted.
SET_SUFFIX='_Set(N(a(m(e)?)?)?)?$'

mapfile -t ALL_BASES < <(
  csvcut -n "$RAW/$FORCE" \
    | sed -E 's/^[[:space:]]*[0-9]+:[[:space:]]*//' \
    | grep -E "$SET_SUFFIX" \
    | sed -E "s/${SET_SUFFIX}//" \
    | sort -u
)

if [[ ${#ALL_BASES[@]} -eq 0 ]]; then
  echo "ERROR: no SetName columns found in $RAW/$FORCE." >&2
  echo "       Expected column names like '<sample>_SetName' (possibly truncated" >&2
  echo "       to '_SetNam' / '_SetNa' / '_SetN'). Check the header with:" >&2
  echo "         csvcut -n $RAW/$FORCE | head" >&2
  exit 1
fi
echo "Found ${#ALL_BASES[@]} samples in $FORCE."

# Take the first N. To pick specific samples instead, set BASES by hand, e.g.
#   BASES=("010322 Sample A1 Data" "010322 Sample B2 Data")
BASES=("${ALL_BASES[@]:0:$N}")

echo "Subsetting to $N sample(s):"
printf '  - %s\n' "${BASES[@]}"

# ---------------------------------------------------------------------
# 2. Subset each compiled file to Time_S + the chosen samples' columns.
# ---------------------------------------------------------------------
# Builds a comma-separated list of exact column names for csvcut. Column names
# containing a comma would break this; none do here, but the Python fallback in
# make_test_data.py handles that case if it ever arises.
subset_file () {
  local infile="$1" outfile="$2"
  local cols="Time_S"

  # All column names in this file, one per line.
  local -a all
  mapfile -t all < <(
    csvcut -n "$infile" | sed -E 's/^[[:space:]]*[0-9]+:[[:space:]]*//'
  )

  local b c
  for b in "${BASES[@]}"; do
    for c in "${all[@]}"; do
      # Match "<base>_<something>" exactly, i.e. this sample's columns only.
      if [[ "$c" == "${b}_"* ]]; then
        cols="${cols},${c}"
      fi
    done
  done

  csvcut -c "$cols" "$infile" > "$outfile"
  echo "  wrote $outfile ($(csvcut -n "$outfile" | wc -l) columns, $(( $(wc -l < "$outfile") - 1 )) rows)"
}

echo "Writing subsetted data files:"
subset_file "$RAW/$FORCE" "$OUT/$FORCE"
subset_file "$RAW/$DISP"  "$OUT/$DISP"
subset_file "$RAW/$SIZE"  "$OUT/$SIZE"

# ---------------------------------------------------------------------
# 3. Subset the metadata to the matching rows.
# ---------------------------------------------------------------------
# Match on FileName, which is unique per recording. Matching on
# (Date_ID, Sample_ID, Replicate) is lossy: '...C3Data' and '...C3.2Data' share
# the triple (210330, C, 3), so each base would pull BOTH rows and the subset
# would end up with duplicate metadata.
#
# csvgrep -m matches a LITERAL string, so sample names need no regex escaping.
# That matters here: base names contain '.' (e.g. 'C3.2Data'), which would be a
# regex wildcard under -r. The row count is checked afterwards.
head -1 "$RAW/$META" > "$OUT/$META"

for b in "${BASES[@]}"; do
  csvgrep -c "FileName" -m "${b}.csv" "$RAW/$META" | tail -n +2 >> "$OUT/$META" || true
done

meta_rows=$(( $(wc -l < "$OUT/$META") - 1 ))
echo "  wrote $OUT/$META ($meta_rows rows)"

if [ "$meta_rows" -eq 0 ]; then
  echo "WARNING: no metadata rows matched. Does $RAW/$META have a FileName column?" >&2
  echo "         Check with: csvcut -n $RAW/$META" >&2
elif [ "$meta_rows" -ne "${#BASES[@]}" ]; then
  echo "WARNING: $meta_rows metadata rows for ${#BASES[@]} samples - expected one each." >&2
  echo "         Duplicate FileName rows in $RAW/$META? Check with:" >&2
  echo "         csvcut -c FileName $RAW/$META | sort | uniq -d" >&2
fi

# ---------------------------------------------------------------------
# 4. Report sizes and next steps.
# ---------------------------------------------------------------------
echo
echo "Test data written to $OUT/:"
du -h "$OUT"/*.csv | sed 's/^/  /'
echo
echo "Run the pipeline against it with:"
echo "  python 01_biomechanical_processing.py --raw-dir $OUT --results-dir results/test -v"
echo "  python 01_biomechanical_processing_revised.py --raw-dir $OUT --results-dir results/test -v"
