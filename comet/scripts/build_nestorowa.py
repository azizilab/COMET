"""Build the nestorowa dataset directory (full Nestorowa et al. 2016 mouse HSPC set, GSE81682, 1654 QC-passed cells).

Raw inputs (put them in RAW):
  GSE81682_HTSeq_counts.txt.gz                     GEO GSE81682
  all_cell_types.txt, coordinates_gene_counts_flow_cytometry.txt.gz
                                                   http://blood.stemcells.cam.ac.uk/single_cell_atlas.html
  BEELINE-data/inputs/scRNA-Seq/mHSC-{E,GM,L}/PseudoTime.csv
                                                   BEELINE data, Zenodo 3701939 (validation pseudotime only)
The ATAC and GRN prior are the paul15 ones (mouse sciATAC atlas; same species and tissue), symlinked from PRIOR.

    python scripts/build_nestorowa.py RAW OUT PRIOR      # needs mygene (Ensembl -> symbol)

Then train Decipher on OUT/rna.h5ad (decipher env) and save OUT/decipher_embeddings.npz:

    a = sc.read_h5ad("OUT/rna.h5ad"); a.X = a.layers["raw_count_int"].copy(); sc.pp.filter_genes(a, min_cells=10)
    t = a.copy(); sc.pp.normalize_total(t); sc.pp.log1p(t); sc.pp.highly_variable_genes(t, n_top_genes=5000)
    a = a[:, t.var.highly_variable].copy(); dc.tl.decipher_train(a, dc.tl.DecipherConfig(), plot_every_k_epochs=-1)
    np.savez("OUT/decipher_embeddings.npz", v=a.obsm["decipher_v"], z=a.obsm["decipher_z"])

obs: cell_type (broad sort gates; "unassigned" otherwise), stage (LT-HSC 0, MPP 1, LMPP/CMP 2, GMP/MEP 3), sort_gate,
ESLAM, facs_*, beeline_pt_{E,GM,L} and pseudotime (their mean). All are used for validation / plots only.
"""
import gzip, os, re, sys
import numpy as np, pandas as pd, anndata as ad

RAW, OUT, PRIOR = sys.argv[1:4]
os.makedirs(OUT, exist_ok=True)

# Ensembl gene ID -> symbol (the counts are indexed by Ensembl ID)
sym_csv = f"{RAW}/ensembl_symbol_mm.csv"
if not os.path.exists(sym_csv):
    import mygene
    ids = [l.split("\t", 1)[0] for l in gzip.open(f"{RAW}/GSE81682_HTSeq_counts.txt.gz", "rt")][1:]
    ids = [i for i in ids if i.startswith("ENSMUSG")]
    res = mygene.MyGeneInfo().querymany(ids, scopes="ensembl.gene", fields="symbol", species="mouse",
                                        as_dataframe=True, verbose=False)
    res = res[~res.index.duplicated(keep="first")]["symbol"].dropna()
    pd.DataFrame({"ensembl": res.index, "symbol": res.values}).to_csv(sym_csv, index=False)


qc = pd.read_csv(f"{RAW}/coordinates_gene_counts_flow_cytometry.txt.gz", sep="\t", index_col=0,
                 usecols=lambda c: not c.startswith("ENSMUSG"))
fix_ids = lambda idx: idx.str.replace(r"^LT[._]HSC", "LT-HSC", regex=True)
qc.index = fix_ids(qc.index)
raw = pd.read_csv(f"{RAW}/GSE81682_HTSeq_counts.txt.gz", sep="\t", index_col=0)
raw = raw[raw.index.str.startswith("ENSMUSG")]
cells = [c for c in qc.index if c in raw.columns]
print(f"QC cells {len(qc)}, in counts {len(cells)}")
raw = raw[cells]
m = pd.read_csv(f"{RAW}/ensembl_symbol_mm.csv"); m = dict(zip(m.ensembl, m.symbol))
sym = raw.index.map(m); raw = raw[sym.notna()]; raw.index = sym[sym.notna()]
raw = raw.groupby(level=0).sum()
X = raw.T.astype(np.float32)

obs = pd.DataFrame(index=X.index)
ct = pd.read_csv(f"{RAW}/all_cell_types.txt", sep="\t", index_col=0); ct.index = fix_ids(ct.index)
ct = ct.reindex(obs.index).fillna(0)
lab = np.full(len(obs), "unassigned", object)
for g, name in [("LTHSC", "LT-HSC"), ("MPP", "MPP"), ("LMPP", "LMPP"), ("CMP", "CMP"), ("GMP", "GMP"), ("MEP", "MEP")]:
    lab[ct[f"{g}_broad"].values == 1] = name
obs["cell_type"] = pd.Categorical(lab)
obs["stage"] = pd.Series(lab, index=obs.index).map({"LT-HSC": 0, "MPP": 1, "LMPP": 2, "CMP": 2, "GMP": 3, "MEP": 3}).astype(float)
obs["sort_gate"] = [re.sub(r"_\d+$", "", c) for c in obs.index]
obs["ESLAM"] = ct["ESLAM"].values.astype(int)
for c in ["FSC.H", "CD34", "CD16", "c.Kit", "EPCR", "Flk2", "CD150", "CD48", "Lin", "Sca1", "DC1", "DC2", "DC3"]:
    obs["facs_" + c.replace(".", "")] = qc.loc[obs.index, c].values
pts = []
for L in ["E", "GM", "L"]:
    p = pd.read_csv(f"{RAW}/BEELINE-data/inputs/scRNA-Seq/mHSC-{L}/PseudoTime.csv", index_col=0)["PseudoTime"]
    p.index = fix_ids(p.index)
    obs[f"beeline_pt_{L}"] = p.reindex(obs.index).values; pts.append(obs[f"beeline_pt_{L}"])
obs["pseudotime"] = pd.concat(pts, axis=1).mean(1).values   # validation only; NaN for cells in no BEELINE lineage

a = ad.AnnData(X=X.values, obs=obs, var=pd.DataFrame(index=X.columns.astype(str)))
a.layers["raw_count_int"] = a.X.copy()
a.write_h5ad(f"{OUT}/rna.h5ad")
print(a); print(obs.cell_type.value_counts().to_dict())
print("BEELINE coverage:", {L: int(obs[f'beeline_pt_{L}'].notna().sum()) for L in ['E','GM','L']})
for f in ["peak_tf.mtx", "tf_names.txt", "peak_gene.csv", "peaks.txt", "atac_matrix.trimmed.mtx.gz", "atac_matrix.trimmed.peaks.txt"]:
    dst = f"{OUT}/{f}"
    if not os.path.exists(dst): os.symlink(f"{PRIOR}/{f}", dst)
