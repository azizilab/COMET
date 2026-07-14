"""
Build COMET "universe" input files for the Paul et al. 2015 mouse
hematopoiesis dataset from CellOracle's precomputed mouse sciATAC-seq
atlas base GRN (Cusanovich et al. 2018), instead of scanning motifs
and linking peaks to genes from scratch.

`co.data.load_mouse_scATAC_atlas_base_GRN()` returns one row per
peak-gene link (peak_id, gene_short_name) plus a 0/1 motif-hit column
for each of 1093 TFs. That is exactly the join COMET's own
`build_M` performs (peak_gene.T @ peak_tf), just staged one step
upstream — so it can be decomposed straight back into COMET's
peaks.txt / peak_gene.csv / peak_tf.mtx / tf_names.txt format.

Verified before writing this:
  - 0 duplicate (peak_id, gene_short_name) rows
  - TF columns are strictly {0, 1} (not motif scores)
  - TF-hit pattern is identical across every row sharing a peak_id
  - base_GRN's 40,623 peak_ids are a 100% subset of
    atac_matrix.binary.qc_filtered.peaks.txt (same source atlas)

Run with the `celloracle_env3` conda environment (has celloracle installed).
Also writes atac_peak_index_map.tsv + atac_matrix.trimmed.peaks.txt, which
trim_paul15_atac.sh uses to shrink the raw ATAC matrix down to only the
peaks the base GRN actually references.
"""

import celloracle as co
import scipy.io
import scipy.sparse as sp

OUT_DIR = "/home/elicanter/datasets/paul2015data"
RAW_ATAC_PEAKS = f"{OUT_DIR}/atac_matrix.binary.qc_filtered.peaks.txt"

base_grn = co.data.load_mouse_scATAC_atlas_base_GRN()
tf_cols = list(base_grn.columns[2:])

# canonical peak order/dedup: first occurrence is sufficient since the
# TF-hit pattern is identical across every row sharing a peak_id.
peaks_unique = base_grn.drop_duplicates(subset="peak_id", keep="first").reset_index(drop=True)
peak_order = peaks_unique["peak_id"].tolist()                # chr_start_end
peak_order_dash = [p.replace("_", "-") for p in peak_order]  # COMET's peaks.txt format

# ── peaks.txt ────────────────────────────────────────────────────────────
with open(f"{OUT_DIR}/peaks.txt", "w") as f:
    f.write("\n".join(peak_order_dash) + "\n")

# ── peak_gene.csv: (peak, gene), one row per peak-gene link ────────────────
peak_gene = base_grn[["peak_id", "gene_short_name"]].rename(
    columns={"peak_id": "peak", "gene_short_name": "gene"}
)
peak_gene["peak"] = peak_gene["peak"].str.replace("_", "-", regex=False)
peak_gene.to_csv(f"{OUT_DIR}/peak_gene.csv", sep="\t", index=False)

# ── peak_tf.mtx (peaks x TFs, binary) + tf_names.txt ────────────────────────
# row order matches peaks.txt / peak_order above
tf_matrix = sp.csr_matrix(peaks_unique[tf_cols].to_numpy(dtype="float32"))
scipy.io.mmwrite(f"{OUT_DIR}/peak_tf.mtx", tf_matrix)

with open(f"{OUT_DIR}/tf_names.txt", "w") as f:
    f.write("\n".join(tf_cols) + "\n")

print(f"peaks.txt:     {len(peak_order_dash)} peaks")
print(f"peak_gene.csv: {len(peak_gene)} links, {peak_gene['gene'].nunique()} genes")
print(f"peak_tf.mtx:   {tf_matrix.shape}, {int(tf_matrix.sum())} hits")
print(f"tf_names.txt:  {len(tf_cols)} TFs")

# ── prep for trimming the raw ATAC matrix down to these peaks ──────────────
# Row numbers are 1-indexed to match MatrixMarket convention, i.e. the raw
# peaks file's line order, which is what the .mtx rows are indexed by.
keep_peaks = set(peak_order)  # underscore format, matches the raw atac peaks file directly

with open(RAW_ATAC_PEAKS) as f:
    f.readline()  # "Feature" header
    raw_peaks = [line.strip() for line in f]

new_idx = 0
with open(f"{OUT_DIR}/atac_peak_index_map.tsv", "w") as map_f, \
     open(f"{OUT_DIR}/atac_matrix.trimmed.peaks.txt", "w") as peaks_f:
    peaks_f.write("Feature\n")
    for old_idx, peak in enumerate(raw_peaks, start=1):
        if peak in keep_peaks:
            new_idx += 1
            map_f.write(f"{old_idx}\t{new_idx}\n")
            peaks_f.write(f"{peak}\n")

print(f"atac trim map: keeping {new_idx} of {len(raw_peaks)} raw ATAC peaks")
print("Run trim_paul15_atac.sh next to produce atac_matrix.trimmed.mtx.gz")
