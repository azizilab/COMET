import pandas as pd
import re

# ── 1. Load file ──────────────────────────────────────────────────────────────
df = pd.read_csv(
    "motif_scanning/homer_motif_results/peak_motif_annotation.tsv",
    sep="\t",
    low_memory=False,
)

# First column is PeakID (name contains the full cmd string)
peak_col = df.columns[0]
df["peak_id"] = df[peak_col].astype(str)

# ── 2. Identify motif columns ─────────────────────────────────────────────────
motif_cols = [c for c in df.columns if "Distance From Peak" in c]
print(f"Found {len(motif_cols)} motif columns")

# ── 3. Parse motif name from column header ────────────────────────────────────
# Column format:  "1-ACTTCCTSTT Distance From Peak(sequence,strand,conservation)"
#                 "1-ACTTCCTB Distance From Peak(...)"   (8-mer variant)
# We extract the sequence/motif identifier that comes right after "N-"
def parse_motif_name(col_name):
    m = re.match(r"^(\d+)-(\S+)\s+Distance From Peak", col_name)
    if m:
        return f"{m.group(1)}-{m.group(2)}"   # e.g. "1-ACTTCCTSTT"
    return col_name  # fallback: keep full column name

motif_name_map = {c: parse_motif_name(c) for c in motif_cols}

# ── 4. Melt to long format ────────────────────────────────────────────────────
long = (
    df[["peak_id"] + motif_cols]
    .melt(id_vars="peak_id", var_name="motif_col", value_name="hit")
)

# ── 5. Keep only real hits (non-null, non-empty, contains a number) ───────────
long = long.dropna(subset=["hit"])
long["hit"] = long["hit"].astype(str).str.strip()
long = long[long["hit"] != ""]
long = long[long["hit"].str.contains(r"\d", na=False)]

# ── 6. Map column name → motif label ─────────────────────────────────────────
long["motif_id"] = long["motif_col"].map(motif_name_map)

# ── 7. (Optional) parse distance from the hit value ──────────────────────────
# HOMER hit format: "chr1,+,45"  or just a distance integer
# Uncomment if you want distance:
# long["distance"] = long["hit"].str.extract(r"(-?\d+)$")[0].astype(float)

# ── 8. Save ───────────────────────────────────────────────────────────────────
out = long[["peak_id", "motif_id"]].drop_duplicates()
out.columns = ["peak_id", "tf_name"]   # rename to match expected output name

out.to_csv("homer_outputs/peak_motifs.csv", index=False)
print(f"Saved {len(out):,} peak–TF pairs")
print(out.head(10))
