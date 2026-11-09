#!/usr/bin/env bash
#
# Shrinks the raw mouse sciATAC-seq atlas matrix
# (atac_matrix.binary.qc_filtered.mtx.gz, 436,206 peaks x 81,173 cells,
# ~422M nonzeros) down to just the 40,623 peaks referenced by CellOracle's
# base GRN, using atac_peak_index_map.tsv from build_paul15_universe.py.
#
# Streams through awk (single pass, small hash of ~40k kept rows) instead
# of loading the full matrix into memory.
#
# Run build_paul15_universe.py first to generate atac_peak_index_map.tsv
# and atac_matrix.trimmed.peaks.txt.
#
# Usage: bash trim_paul15_atac.sh DIR   (DIR holds the raw matrix and the index map; output is written there)

set -euo pipefail

DIR="${1:?usage: trim_paul15_atac.sh DIR}"
RAW_MTX="$DIR/atac_matrix.binary.qc_filtered.mtx.gz"
MAP="$DIR/atac_peak_index_map.tsv"
BODY="$DIR/.atac_trimmed_body.tmp"
STATS="$DIR/.atac_trim_stats.tmp"

zcat "$RAW_MTX" | awk -v mapfile="$MAP" -v body="$BODY" '
    BEGIN {
        while ((getline line < mapfile) > 0) {
            split(line, a, "\t")
            keep[a[1]] = a[2]
        }
        n_out = 0
    }
    NR == 2 { n_cells = $2; next }   # dims line: n_peaks n_cells nnz
    NR <= 2 { next }                  # MatrixMarket banner line
    ($1 in keep) {
        print keep[$1], $2 > body
        n_out++
    }
    END { print n_cells, n_out > "'"$STATS"'" }
'

read N_CELLS NNZ < "$STATS"
N_PEAKS=$(wc -l < "$MAP")

{
    echo "%%MatrixMarket matrix coordinate pattern general"
    echo "$N_PEAKS $N_CELLS $NNZ"
    cat "$BODY"
} | gzip > "$DIR/atac_matrix.trimmed.mtx.gz"

rm -f "$BODY" "$STATS"
echo "Wrote atac_matrix.trimmed.mtx.gz: $N_PEAKS peaks x $N_CELLS cells, $NNZ nonzeros"
