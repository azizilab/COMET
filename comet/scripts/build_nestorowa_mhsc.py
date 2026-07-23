"""
Build a COMET dataset directory for a BEELINE mHSC lineage (Nestorowa et al.
2016 mouse HSPCs, GSE81682), for benchmarking COMET's GRN inference against
BEELINE's gold-standard reference networks.

Why this exists
---------------
BEELINE ships mHSC-E/GM/L as *log-normalized* expression + a Slingshot
PseudoTime, keyed by the original Nestorowa cell IDs (HSPC_###, LT-HSC_###,
Prog_###). COMET (and Decipher) need *raw counts*: data/load.py's
build_universe runs sc.pp.normalize_total + log1p itself, so feeding it
BEELINE's already-normalized values would double-normalize. So we recover raw
counts from GSE81682, subset to exactly the BEELINE cells for the chosen
lineage, and map Ensembl IDs -> gene symbols so var_names align with the mouse
ATAC prior (peak_gene "gene" column is a gene short name, e.g. Gata1).

The scATAC prior is *reused unchanged* from the Paul 2015 setup: it is the
CellOracle/Cusanovich mouse sciATAC atlas base GRN (genome-wide, not
cell-matched), so it serves any mouse RNA dataset. We just symlink those files.

Outputs (under datasets/nestorowa_mhsc_<lineage>/)
--------------------------------------------------
  Nestorowa_mHSC_<L>.h5ad   cells x genes; .X and layers['raw_count_int'] are
                            raw integer counts; obs has 'cell_type' and
                            'PseudoTime'; uns['beeline_genes'] lists the genes
                            in BEELINE's expression panel (for panel-restricted
                            scoring).
  peak_tf.mtx, tf_names.txt, peak_gene.csv, peaks.txt,
  atac_matrix.trimmed.mtx.gz, atac_matrix.trimmed.peaks.txt
                            symlinks to ../paul2015data/ (reused mouse prior)

decipher_embeddings.npz is produced separately by comet/nestorowa_decipher.ipynb.

Run with the decipher-env conda env (has scanpy + anndata). mygene is only in
celloracle_env3, so the Ensembl->symbol map is fetched once and cached to
datasets/_beeline_raw/ensembl_symbol_mm.csv; if that cache is missing and
mygene is unavailable, re-run this script under celloracle_env3 once to build it.

Usage:  python build_nestorowa_mhsc.py --lineage E     # E | GM | L
"""

import argparse
import os
import re

import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad

RAW_DIR = "/home/elicanter/datasets/_beeline_raw"
GSE = f"{RAW_DIR}/GSE81682_HTSeq_counts.txt.gz"
PRIOR_DIR = "/home/elicanter/datasets/paul2015data"          # reused mouse ATAC prior
MAP_CACHE = f"{RAW_DIR}/ensembl_symbol_mm.csv"

# ATAC prior files reused verbatim across any mouse dataset
PRIOR_FILES = [
    "peak_tf.mtx", "tf_names.txt", "peak_gene.csv", "peaks.txt",
    "atac_matrix.trimmed.mtx.gz", "atac_matrix.trimmed.peaks.txt",
]


def ensembl_to_symbol(ensembl_ids: list[str]) -> dict:
    """Map mouse Ensembl gene IDs -> symbols, cached to disk.

    Uses the cache if present; otherwise queries mygene (celloracle_env3).
    """
    if os.path.exists(MAP_CACHE):
        m = pd.read_csv(MAP_CACHE)
        return dict(zip(m["ensembl"].astype(str), m["symbol"].astype(str)))

    import mygene  # only available in celloracle_env3
    mg = mygene.MyGeneInfo()
    res = mg.querymany(
        ensembl_ids, scopes="ensembl.gene", fields="symbol",
        species="mouse", as_dataframe=True, df_index=True,
    )
    res = res[~res.index.duplicated(keep="first")]
    mapping = res["symbol"].dropna().to_dict()
    pd.DataFrame(
        {"ensembl": list(mapping.keys()), "symbol": list(mapping.values())}
    ).to_csv(MAP_CACHE, index=False)
    return mapping


def main(lineage: str):
    lineage = lineage.upper()
    be_dir = f"{RAW_DIR}/BEELINE-data/inputs/scRNA-Seq/mHSC-{lineage}"
    if not os.path.isdir(be_dir):
        # only mHSC-E is extracted by default; extract others on demand
        import zipfile
        with zipfile.ZipFile(f"{RAW_DIR}/BEELINE-data.zip") as z:
            for f in ("ExpressionData.csv", "PseudoTime.csv", "GeneOrdering.csv"):
                z.extract(f"BEELINE-data/inputs/scRNA-Seq/mHSC-{lineage}/{f}", RAW_DIR)

    out_dir = f"/home/elicanter/datasets/nestorowa_mhsc_{lineage.lower()}"
    os.makedirs(out_dir, exist_ok=True)

    # --- BEELINE cell set + panel + pseudotime -----------------------------
    be_expr = pd.read_csv(f"{be_dir}/ExpressionData.csv", index_col=0)   # genes x cells
    be_cells = list(be_expr.columns)
    be_panel_genes = [g.upper() for g in be_expr.index]                  # scoring panel
    pt = pd.read_csv(f"{be_dir}/PseudoTime.csv", index_col=0)["PseudoTime"]
    print(f"mHSC-{lineage}: {len(be_cells)} cells, {len(be_panel_genes)} panel genes")

    # --- raw counts (Ensembl x cells) --------------------------------------
    raw = pd.read_csv(GSE, sep="\t", index_col=0)                        # genes(ENS) x cells
    raw.columns = [c.replace("-", "_") for c in raw.columns]            # LT-HSC -> LT_HSC
    missing = [c for c in be_cells if c not in raw.columns]
    assert not missing, f"{len(missing)} BEELINE cells absent from GSE81682, e.g. {missing[:5]}"
    raw = raw[be_cells]                                                  # subset + order to BEELINE
    # drop HTSeq summary rows (e.g. __no_feature) if present
    raw = raw[~raw.index.astype(str).str.startswith("__")]

    # --- Ensembl -> symbol --------------------------------------------------
    mapping = ensembl_to_symbol(list(raw.index.astype(str)))
    symbols = pd.Series(raw.index.astype(str)).map(mapping)
    keep = symbols.notna().values
    raw = raw.loc[keep]
    symbols = symbols[keep].values
    # collapse genes sharing a symbol by summing counts
    raw.index = symbols
    raw = raw.groupby(level=0).sum()
    print(f"mapped to {raw.shape[0]} unique gene symbols")

    # --- assemble AnnData (cells x genes, raw counts) ----------------------
    X = raw.T                                                            # cells x genes
    adata = ad.AnnData(
        X=X.values.astype(np.float32),
        obs=pd.DataFrame(index=X.index),
        var=pd.DataFrame(index=X.columns.astype(str)),
    )
    adata.layers["raw_count_int"] = adata.X.copy()
    adata.obs["PseudoTime"] = pt.reindex(adata.obs_names).values
    adata.obs["cell_type"] = [re.sub(r"_\d+$", "", c) for c in adata.obs_names]
    adata.obs["cell_type"] = adata.obs["cell_type"].astype("category")
    # genes from BEELINE's panel that survived symbol mapping (for scoring)
    panel_present = [g for g in be_panel_genes
                     if g in set(adata.var_names.str.upper())]
    adata.uns["beeline_genes"] = np.array(panel_present, dtype=object)
    adata.uns["beeline_lineage"] = f"mHSC-{lineage}"

    out_h5ad = f"{out_dir}/Nestorowa_mHSC_{lineage}.h5ad"
    adata.write_h5ad(out_h5ad)
    print(f"wrote {out_h5ad}: {adata.n_obs} cells x {adata.n_vars} genes")
    print(f"  cell_type counts: {adata.obs['cell_type'].value_counts().to_dict()}")
    print(f"  BEELINE panel genes present: {len(panel_present)} / {len(be_panel_genes)}")

    # --- symlink reused mouse ATAC prior -----------------------------------
    for f in PRIOR_FILES:
        src, dst = f"{PRIOR_DIR}/{f}", f"{out_dir}/{f}"
        if os.path.islink(dst) or os.path.exists(dst):
            os.remove(dst)
        os.symlink(src, dst)
    print(f"symlinked {len(PRIOR_FILES)} mouse ATAC prior files into {out_dir}")
    print("Next: run comet/nestorowa_decipher.ipynb to write decipher_embeddings.npz")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--lineage", default="E", choices=["E", "GM", "L"])
    main(ap.parse_args().lineage)
