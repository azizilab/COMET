BASE = "/home/elicanter/datasets/hematopoiesis/"

DATA = {
    "rna": f"{BASE}/velten_rna.h5ad",
    "atac": f"{BASE}/GSE129785_scATAC-Hematopoiesis-CD34.mtx.gz",
    "atac_peaks": f"{BASE}/GSE129785_scATAC-Hematopoiesis-CD34.peaks.txt.gz",
    "peak_tf": f"{BASE}/peak_tf.mtx",
    "tf_names": f"{BASE}/tf_names.txt",
    "peak_gene": f"{BASE}/peak_gene.csv",
    "peaks": f"{BASE}/peaks.txt",
}

MODEL = {
    "D_ATOMS": 4,
    "hidden": 64,
}

TRAIN = {
    "batch_size": 512,
    "lr_grn": 5e-4,
    "lr_vel": 2e-4,
    "epochs": 120,
}