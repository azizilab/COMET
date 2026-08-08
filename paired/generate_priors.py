import os
import pickle
import anndata as ad
import pandas as pd
import numpy as np
import scipy.io
from scipy.sparse import lil_matrix
import celloracle as co
import genomepy

# Target files
CSV_OUT = "ma2020_peak_gene.csv"
TSS_PICKLE = "ma2020_tss_dict.pkl"
SCAN_CACHE = "ma2020_scan_df.pkl"
MOTIF_CACHE = "ma2020_motif_dict.pkl"
MTX_OUT = "ma2020_peak_tf.mtx"
TXT_OUT = "ma2020_tf_names.txt"

# --- CHECKPOINT 1: Genome Download ---
if not os.path.exists(os.path.expanduser("~/.local/share/genomes/mm10")):
    print("Downloading mm10 genome...")
    genomepy.install_genome("mm10", "UCSC")
else:
    print("mm10 genome already found locally.")

# Load ATAC and Format Peaks
print("Loading ATAC h5ad to extract peak list...")
atac = ad.read_h5ad("Ma-2020-ATAC.h5ad")
peaks = atac.var['chrom'].astype(str) + "_" + atac.var['chromStart'].astype(str) + "_" + atac.var['chromEnd'].astype(str)
peak_list = peaks.tolist()
peak_to_idx = {p: i for i, p in enumerate(peak_list)}

# --- CHECKPOINT 2: Peak-to-Gene Linkage ---
if os.path.exists(TSS_PICKLE):
    print("Found cached TSS mapping! Loading from pickle...")
    with open(TSS_PICKLE, "rb") as f:
        tss_obj = pickle.load(f)
else:
    print("Mapping peaks to mm10 genes based on Transcription Start Sites...")
    tss_obj = co.motif_analysis.get_tss_info(peak_str_list=peak_list, ref_genome="mm10")
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
        peak_strings = tss_obj[c_chr].astype(str) + "_" + tss_obj["start"].astype(str) + "_" + tss_obj["end"].astype(str)
        peak_gene_df = pd.DataFrame({"peak": peak_strings, "gene": tss_obj[gene_col].astype(str)})
    else:
        raise ValueError(f"Unrecognized DataFrame structure. Columns: {tss_obj.columns.tolist()}")
else:
    raise TypeError(f"Unrecognized CellOracle return type: {type(tss_obj)}")

peak_gene_df = peak_gene_df.dropna().drop_duplicates()
peak_gene_df.to_csv(CSV_OUT, sep="\t", index=False)
print(f" -> Saved {CSV_OUT} ({len(peak_gene_df)} valid regulatory links found)")

# --- CHECKPOINT 3: Motif Scanning ---
if os.path.exists(SCAN_CACHE) and os.path.exists(MOTIF_CACHE):
    print("Found cached motif scan results! Bypassing the 14-minute scan...")
    scan_df = pd.read_pickle(SCAN_CACHE)
    with open(MOTIF_CACHE, "rb") as f:
        motif_to_tfs = pickle.load(f)
else:
    print("Scanning sequences for TF motifs (this will take several minutes)...")
    co_formatted_df = pd.DataFrame({
        "peak_id": peak_gene_df["peak"],
        "gene_short_name": peak_gene_df["gene"]
    })

    tfi = co.motif_analysis.TFinfo(peak_data_frame=co_formatted_df, ref_genome="mm10")
    tfi.scan(fpr=0.02, motifs=None) 
    
    scan_df = tfi.scanned_df
    motif_to_tfs = tfi.dic_motif2TFs
    
    print("Caching scan results to disk so we never have to run this step again...")
    scan_df.to_pickle(SCAN_CACHE)
    with open(MOTIF_CACHE, "wb") as f:
        pickle.dump(motif_to_tfs, f)

# --- CHECKPOINT 4: Matrix Construction ---
print("Building Peak x TF matrix dynamically...")

# 1. Identify Peak Column/Index robustly
peak_vals = None

# Check if index itself contains chromosome coordinates
if scan_df.index.astype(str).str.contains("chr").any():
    print(" -> Peak coordinates detected in scan_df index.")
    peak_vals = scan_df.index.astype(str).values
else:
    # Check explicit column names (excluding row index numbers)
    for col in scan_df.columns:
        if col.lower() in ['seq_id', 'seq', 'peak_id', 'peak', 'sequence']:
            peak_vals = scan_df[col].astype(str).values
            print(f" -> Peak coordinates detected in column '{col}'.")
            break
            
    # Fallback: scan columns for string contents matching 'chr'
    if peak_vals is None:
        for col in scan_df.columns:
            if scan_df[col].astype(str).str.contains("chr").any():
                peak_vals = scan_df[col].astype(str).values
                print(f" -> Peak coordinates detected via content in column '{col}'.")
                break

if peak_vals is None:
    raise ValueError(f"Could not identify peak coordinates in scan_df! Columns: {scan_df.columns.tolist()}")

# 2. Identify Motif Column robustly
motif_vals = None
for col in scan_df.columns:
    if col.lower() in ['motif_id', 'motif', 'motifs']:
        motif_vals = scan_df[col].astype(str).values
        print(f" -> Motifs detected in column '{col}'.")
        break

if motif_vals is None:
    # Check if motif IDs are in index or fallback search
    for col in scan_df.columns:
        if scan_df[col].astype(str).str.contains("GM\.").any():
            motif_vals = scan_df[col].astype(str).values
            print(f" -> Motifs detected via content in column '{col}'.")
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
    peak = peak_vals[i].replace(":", "_").replace("-", "_")  # Standardize delimiters
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
print("Done! You are ready to run the pipeline.")
