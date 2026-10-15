import os
import argparse
import pickle
import pandas as pd
import numpy as np
import scipy.io
from scipy.sparse import lil_matrix
import celloracle as co
import genomepy


def load_peak_list(path):
    """Read a peak list from a text file -- no .h5ad required. Accepts either the canonical
    `atac_peaks.txt` format (tab-separated, header row, a `Feature` column -- exactly the file
    COMET's own atac_peaks_file= contract requires, see README_DATASETS.md) or a plain
    headerless single-column peak list (e.g. an existing peaks.txt). Returns peak strings as-is
    (format sanitized by the caller)."""
    df = pd.read_csv(path, sep="\t")
    if "Feature" in df.columns:
        col = df["Feature"]
    elif df.shape[1] == 1:
        col = pd.read_csv(path, sep="\t", header=None)[0]
    else:
        raise ValueError(f"{path}: expected a 'Feature' column (like atac_peaks.txt) or a single "
                          f"unheadered column of peak strings; got columns {df.columns.tolist()}")
    return col.astype(str).tolist()


def to_underscore(peak):
    """Normalize a peak string to CellOracle's chr_start_end convention, whatever separator
    style it arrived in (chr:start-end, chr-start-end, or already chr_start_end)."""
    return peak.replace(":", "_").replace("-", "_")


def to_dash(peak_underscore):
    """chr_start_end -> chr-start-end (COMET's peak_gene.csv/peaks.txt convention, see
    comet/data/load.py's `fix()` step). Split from the right so a chrom name that legitimately
    contains underscores (e.g. unplaced scaffolds like "chr1_KI270706v1_random") isn't corrupted
    -- only the trailing start/end fields are dash-joined."""
    parts = peak_underscore.rsplit("_", 2)
    if len(parts) != 3:
        raise ValueError(f"Could not parse peak string into chrom/start/end: {peak_underscore!r}")
    return "-".join(parts)


def main():
    parser = argparse.ArgumentParser(description="Generate Base GRN Priors from scATAC-seq data")
    parser.add_argument("--input", required=True,
                         help="Path to a peak list text file -- normally your dataset's own "
                              "atac_peaks.txt (tab-separated, header row, 'Feature' column; the "
                              "same file COMET's atac_peaks_file= contract requires, so you "
                              "don't need to build a separate .h5ad just for this). A plain "
                              "headerless single-column peak list also works.")
    parser.add_argument("--genome", required=True,
                         help="Reference genome, any genomepy/UCSC assembly name CellOracle "
                              "supports (e.g. mm10, mm39 for mouse; hg19, hg38 for human).")
    parser.add_argument("--outdir", required=True,
                         help="Dataset base directory to write the 4 canonical GRN-prior files "
                              "into: peak_gene.csv, peak_tf.mtx, tf_names.txt, peaks.txt. Pass "
                              "the same directory you'll use as base= for COMET.")
    args = parser.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    # Final canonical files -- exact names load_generic() expects inside `base` (see
    # README_DATASETS.md); no prefix, so this dir can be passed straight to COMET as `base=`.
    CSV_OUT = os.path.join(args.outdir, "peak_gene.csv")
    MTX_OUT = os.path.join(args.outdir, "peak_tf.mtx")
    TXT_OUT = os.path.join(args.outdir, "tf_names.txt")
    PEAKS_OUT = os.path.join(args.outdir, "peaks.txt")
    # Intermediate caches (resumable checkpoints) -- distinct names, never collide with the above.
    TSS_PICKLE = os.path.join(args.outdir, ".cache_tss_dict.pkl")
    SCAN_CACHE = os.path.join(args.outdir, ".cache_scan_df.pkl")
    MOTIF_CACHE = os.path.join(args.outdir, ".cache_motif_dict.pkl")

    # --- CHECKPOINT 1: Genome Download ---
    genome_path = os.path.expanduser(f"~/.local/share/genomes/{args.genome}")
    if not os.path.exists(genome_path):
        print(f"Downloading {args.genome} genome...")
        genomepy.install_genome(args.genome, "UCSC")
    else:
        print(f"{args.genome} genome already found locally.")

    # Load peak list directly from text -- no .h5ad needed, this file already exists as part of
    # COMET's own canonical dataset contract (atac_peaks.txt).
    print(f"Loading peak list from {args.input}...")
    peaks_raw = load_peak_list(args.input)
    if not peaks_raw:
        raise ValueError(f"{args.input}: no peaks found")

    # CellOracle's peak_str_list convention is underscore-joined "chr_start_end" -- keep this
    # format for every celloracle-facing call below, regardless of the input file's separator.
    peak_list = [to_underscore(p) for p in peaks_raw]
    peak_to_idx = {p: i for i, p in enumerate(peak_list)}
    # COMET's peak_tf.mtx / peak_gene.csv / peaks.txt convention is dash-joined "chr-start-end"
    # (see comet/data/load.py's `fix()` step, which assumes this exact format when matching
    # peak_gene.csv against the dataset's own ATAC peaks file).
    peak_list_canon = [to_dash(p) for p in peak_list]
    underscore_to_canon = dict(zip(peak_list, peak_list_canon))
    print(f" -> Loaded {len(peak_list)} peaks")

    # --- CHECKPOINT 2: Peak-to-Gene Linkage ---
    if os.path.exists(TSS_PICKLE):
        print("Found cached TSS mapping! Loading from pickle...")
        with open(TSS_PICKLE, "rb") as f:
            tss_obj = pickle.load(f)
    else:
        print(f"Mapping peaks to {args.genome} genes based on Transcription Start Sites...")
        tss_obj = co.motif_analysis.get_tss_info(peak_str_list=peak_list, ref_genome=args.genome)
        with open(TSS_PICKLE, "wb") as f:
            pickle.dump(tss_obj, f)

    # Auto-detect parser for regulatory links
    print("Formatting regulatory links dynamically...")
    records = []

    if isinstance(tss_obj, dict):
        first_key = list(tss_obj.keys())[0]
        key_is_peak = "_" in first_key and first_key.startswith("chr")
        for k, v in tss_obj.items():
            v_list = v if isinstance(v, (list, tuple, np.ndarray)) else [v]
            for item in v_list:
                if pd.isna(item): continue
                if key_is_peak:
                    records.append({"peak": str(k), "gene": str(item)})
                else:
                    records.append({"peak": str(item), "gene": str(k)})
        peak_gene_df = pd.DataFrame(records)
    elif isinstance(tss_obj, pd.DataFrame):
        cols = [c.lower() for c in tss_obj.columns]
        gene_col = next((c for c in tss_obj.columns if "gene" in c.lower()), None)
        peak_col = next((c for c in tss_obj.columns if "peak" in c.lower() or "seq" in c.lower()), None)
        
        if peak_col and gene_col:
            peak_gene_df = pd.DataFrame({"peak": tss_obj[peak_col].astype(str), "gene": tss_obj[gene_col].astype(str)})
        elif gene_col and any(c in cols for c in ["chrom", "chr"]):
            c_chr = "chrom" if "chrom" in cols else "chr"
            s = tss_obj["start"].astype(float).astype(int).astype(str)
            e = tss_obj["end"].astype(float).astype(int).astype(str)
            peak_strings = tss_obj[c_chr].astype(str) + "_" + s + "_" + e
            peak_gene_df = pd.DataFrame({"peak": peak_strings, "gene": tss_obj[gene_col].astype(str)})
        else:
            raise ValueError(f"Unrecognized DataFrame structure. Columns: {tss_obj.columns.tolist()}")
    else:
        raise TypeError(f"Unrecognized CellOracle return type: {type(tss_obj)}")

    # peak_gene_df stays in celloracle's underscore convention -- it feeds TFinfo() below, which
    # expects that format. Only the on-disk CSV needs COMET's dash convention (see load_generic's
    # `fix()` step), so convert into a separate frame for writing and never mutate peak_gene_df.
    peak_gene_df = peak_gene_df.dropna().drop_duplicates()
    before = len(peak_gene_df)
    peak_gene_out = peak_gene_df.copy()
    peak_gene_out["peak"] = peak_gene_out["peak"].map(underscore_to_canon)
    n_unmapped = peak_gene_out["peak"].isna().sum()
    if n_unmapped:
        print(f"WARNING: dropping {n_unmapped} peak_gene rows whose peak string didn't match "
              f"the input peak list (unexpected -- check get_tss_info output format)")
        peak_gene_out = peak_gene_out.dropna(subset=["peak"])
    peak_gene_out.to_csv(CSV_OUT, sep="\t", index=False)
    print(f" -> Saved {CSV_OUT} ({len(peak_gene_out)}/{before} valid regulatory links found)")

    # peaks.txt: full peak list, canonical dash format, in the exact row order peak_tf.mtx uses.
    with open(PEAKS_OUT, "w") as f:
        f.write("\n".join(peak_list_canon) + "\n")
    print(f" -> Saved {PEAKS_OUT} ({len(peak_list_canon)} peaks, same row order as {os.path.basename(MTX_OUT)})")

    # --- CHECKPOINT 3: Motif Scanning ---
    if os.path.exists(SCAN_CACHE) and os.path.exists(MOTIF_CACHE):
        print("Found cached motif scan results! Bypassing the scan...")
        scan_df = pd.read_pickle(SCAN_CACHE)
        with open(MOTIF_CACHE, "rb") as f:
            motif_to_tfs = pickle.load(f)
    else:
        print("Scanning sequences for TF motifs (this will take several minutes)...")
        co_formatted_df = pd.DataFrame({
            "peak_id": peak_gene_df["peak"],
            "gene_short_name": peak_gene_df["gene"]
        })

        tfi = co.motif_analysis.TFinfo(peak_data_frame=co_formatted_df, ref_genome=args.genome)
        tfi.scan(fpr=0.02, motifs=None) 
        
        scan_df = tfi.scanned_df
        motif_to_tfs = tfi.dic_motif2TFs
        
        print("Caching scan results to disk...")
        scan_df.to_pickle(SCAN_CACHE)
        with open(MOTIF_CACHE, "wb") as f:
            pickle.dump(motif_to_tfs, f)

    # --- CHECKPOINT 4: Matrix Construction ---
    print("Building Peak x TF matrix dynamically...")

    # 1. Identify Peak Column/Index robustly
    peak_vals = None
    if scan_df.index.astype(str).str.contains("chr").any():
        peak_vals = scan_df.index.astype(str).values
    else:
        for col in scan_df.columns:
            if col.lower() in ['seq_id', 'seq', 'peak_id', 'peak', 'sequence']:
                peak_vals = scan_df[col].astype(str).values
                break
        if peak_vals is None:
            for col in scan_df.columns:
                if scan_df[col].astype(str).str.contains("chr").any():
                    peak_vals = scan_df[col].astype(str).values
                    break

    if peak_vals is None:
        raise ValueError(f"Could not identify peak coordinates in scan_df! Columns: {scan_df.columns.tolist()}")

    # 2. Identify Motif Column robustly
    motif_vals = None
    for col in scan_df.columns:
        if col.lower() in ['motif_id', 'motif', 'motifs']:
            motif_vals = scan_df[col].astype(str).values
            break

    if motif_vals is None:
        for col in scan_df.columns:
            if scan_df[col].astype(str).str.contains(r"GM\.").any():
                motif_vals = scan_df[col].astype(str).values
                break

    if motif_vals is None:
        raise ValueError(f"Could not identify motif IDs in scan_df! Columns: {scan_df.columns.tolist()}")

    # Extract all unique TFs from the dictionary
    all_tfs = set()
    for tfs in motif_to_tfs.values():
        if isinstance(tfs, str):
            all_tfs.add(tfs)
        else:
            all_tfs.update(tfs)
    tf_names_list = sorted(list(all_tfs))
    tf_to_idx = {tf: i for i, tf in enumerate(tf_names_list)}

    # Initialize empty sparse matrix [Total ATAC Peaks x Total TFs]
    P = len(peak_list)
    T = len(tf_names_list)
    peak_tf_mat = lil_matrix((P, T), dtype=np.float32)

    print(f"Populating matrix ({P} peaks x {T} TFs) across {len(scan_df)} entries...")

    # Iterate through scanned results and map matches
    matches_found = 0
    for i in range(len(scan_df)):
        peak = peak_vals[i].replace(":", "_").replace("-", "_")  
        motif = motif_vals[i]
        
        if peak in peak_to_idx and motif in motif_to_tfs:
            p_idx = peak_to_idx[peak]
            tfs = motif_to_tfs[motif]
            if isinstance(tfs, str):
                tfs = [tfs]
                
            for tf in tfs:
                if tf in tf_to_idx:
                    peak_tf_mat[p_idx, tf_to_idx[tf]] = 1.0
                    matches_found += 1

    print(f" -> Successfully matched {matches_found} peak-TF interactions!")

    # Save the final matrix and names
    print("Writing matrix to disk...")
    scipy.io.mmwrite(MTX_OUT, peak_tf_mat.tocsr())
    with open(TXT_OUT, "w") as f:
        f.write("\n".join(tf_names_list) + "\n")
        
    print(f" -> Saved {MTX_OUT} and {TXT_OUT}")
    print(f"Done. {args.outdir} now has peak_gene.csv, peak_tf.mtx, tf_names.txt, peaks.txt -- "
          f"ready to use as `base=` for COMET (still need atac_matrix.mtx.gz/atac_peaks.txt, "
          f"rna.h5ad, and decipher_embeddings.npz; see README_DATASETS.md).")

if __name__ == "__main__":
    main()