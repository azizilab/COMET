"""
Convert HOMER annotatePeaks output to peak_motifs.csv, with TF gene symbols that match adata.var_names.

  1. Gene symbols are taken from the motifN.motif headers,
     e.g. "BestGuess:IRF8(IRF)/BMDM..." -> "IRF8", "BestGuess:PB0058.1_Sfpi1_1/Jaspar" -> "Sfpi1";
     unmapped indices fall back to rank N in knownResults.txt.
  2. The 8/10/12-mer length variants of a motif index are collapsed: a peak is a hit if any variant hits.
  3. Motifs hitting more than --max-coverage of all peaks are dropped as uninformative.

Usage:
    python parse_homer_to_tf_names.py \
        --homer-dir motif_scanning/homer_motif_results \
        --output homer_outputs/peak_motifs.csv \
        [--max-coverage 0.60]
"""

import argparse
import os
import re
from collections import defaultdict

import pandas as pd


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def clean_tf_name(raw: str) -> str:
    """
    Extract a clean gene symbol from a HOMER motif name string.

    Handles formats like:
      "BestGuess:IRF8(IRF)/BMDM-IRF8-ChIP-Seq..."  → "IRF8"
      "BestGuess:PB0058.1_Sfpi1_1/Jaspar(0.941)"   → "Sfpi1"
      "BestGuess:AP-1(bZIP)/ThioMac-..."            → "AP1"
      "BestGuess:CEBP:AP1(bZIP)/..."                → "CEBP"
      "ETS1(ETS)/Jurkat-ETS1-ChIP-Seq.../Homer"     → "ETS1"
      "1-ACTTCCTBTTWN,BestGuess:PB0058.1_Sfpi1_1"  → "Sfpi1"
    """
    if "BestGuess:" in raw:
        raw = raw.split("BestGuess:")[1]

    # Strip database ID prefix like "PB0058.1_" or "POL009.1_"
    raw = re.sub(r"^[A-Z]{2,}\d+\.\d+_", "", raw)

    # Take everything before first ( / : \t space
    symbol = re.split(r"[(/:\t\s]", raw)[0].strip()

    # Remove hyphens (AP-1 → AP1)
    symbol = symbol.replace("-", "")

    return symbol


def parse_known_results(known_results_path: str) -> dict:
    """
    Parse knownResults.txt → {rank_str: clean_tf_name}.
    Used as fallback when motifN.motif files don't cover an index.
    """
    rank_to_tf = {}
    with open(known_results_path) as f:
        f.readline()  # skip header
        for i, line in enumerate(f, 1):
            cols = line.strip().split("\t")
            if not cols or not cols[0]:
                continue
            rank_to_tf[str(i)] = clean_tf_name(cols[0])
    print(f"Parsed {len(rank_to_tf)} known TF motifs from knownResults.txt")
    print(f"Top 10: {[rank_to_tf[str(i)] for i in range(1, 11) if str(i) in rank_to_tf]}")
    return rank_to_tf


def parse_homer_motif_files(homer_results_dir: str) -> dict:
    """
    Read motifN.motif files to get the BestGuess TF name for each de novo index.

    Header line of motifN.motif:
      >CONSENSUS\\tN-CONSENSUS,BestGuess:GENENAME(family)/source...\\tlogodds\\t...

    Returns {denovo_index_str: clean_tf_name}
    """
    mapping = {}
    if not os.path.isdir(homer_results_dir):
        print(f"  homerResults/ dir not found: {homer_results_dir}")
        return mapping

    motif_files = sorted([
        f for f in os.listdir(homer_results_dir)
        if re.match(r"^motif\d+\.motif$", f)
    ])
    print(f"Found {len(motif_files)} motifN.motif files in {homer_results_dir}")

    for fname in motif_files:
        fpath = os.path.join(homer_results_dir, fname)
        try:
            with open(fpath) as f:
                header = f.readline().strip()
            if not header.startswith(">"):
                continue
            parts = header[1:].split("\t")
            if len(parts) < 2:
                continue
            name_field = parts[1]  # e.g. "1-ACTTCCTBTTWN,BestGuess:IRF8(...)"

            # Extract de novo index from name field
            idx_match = re.match(r"^(\d+)-", name_field)
            if not idx_match:
                continue
            denovo_idx = idx_match.group(1)

            tf_name = clean_tf_name(name_field)
            mapping[denovo_idx] = tf_name

        except Exception as e:
            print(f"  Warning: could not parse {fname}: {e}")

    print(f"Extracted TF names for {len(mapping)} de novo motif indices")
    return mapping


def build_index_to_tf(homer_results_dir: str, known_rank_to_tf: dict) -> dict:
    """
    Build complete {denovo_index_str: clean_tf_name} mapping.
    Priority: motifN.motif BestGuess > knownResults.txt rank fallback.
    """
    from_motif_files = parse_homer_motif_files(homer_results_dir)

    max_idx = max(
        max((int(k) for k in known_rank_to_tf), default=0),
        max((int(k) for k in from_motif_files), default=0),
    )

    mapping = {}
    for i in range(1, max_idx + 1):
        idx = str(i)
        if idx in from_motif_files:
            mapping[idx] = from_motif_files[idx]
        elif idx in known_rank_to_tf:
            mapping[idx] = known_rank_to_tf[idx]

    print(f"Total index→TF mappings built: {len(mapping)}")
    return mapping


# ---------------------------------------------------------------------------
# Peak annotation parsing
# ---------------------------------------------------------------------------

def parse_peak_annotation(
    tsv_path: str,
    index_to_tf: dict,
    max_coverage: float = 0.60,
) -> pd.DataFrame:
    """
    Parse peak_motif_annotation.tsv and return clean peak-TF pairs.

    Key behaviour:
    - Groups the three length variants (8/10/12-mer) per de novo index.
      A peak has motif N if ANY variant column is non-empty.
      This avoids triple-counting while retaining sensitivity.
    - Drops motifs with coverage > max_coverage (degenerate/noisy).
    - Returns DataFrame [peak_id, tf_name] with clean gene symbols only.
    """
    print(f"\nReading header from {tsv_path} ...")
    with open(tsv_path) as f:
        header_line = f.readline().rstrip("\n")
    all_cols = header_line.split("\t")

    motif_col_pattern = re.compile(r"^(\d+)-\S+\s+Distance From Peak")

    # Group column positions by de novo index
    index_to_col_positions = defaultdict(list)
    unmapped_indices = set()

    for i, col in enumerate(all_cols):
        m = motif_col_pattern.match(col)
        if m:
            idx = m.group(1)
            if idx in index_to_tf:
                index_to_col_positions[idx].append(i)
            else:
                unmapped_indices.add(idx)

    print(f"De novo indices with TF name : {len(index_to_col_positions)}")
    print(f"De novo indices unmapped     : {len(unmapped_indices)}")
    if unmapped_indices:
        print(f"  Unmapped indices: {sorted(unmapped_indices, key=int)}")

    if not index_to_col_positions:
        raise ValueError("No motif columns could be mapped. Check your paths.")

    # Read only peak ID + motif columns
    all_motif_positions = sorted(set(
        pos for positions in index_to_col_positions.values()
        for pos in positions
    ))
    use_cols  = [0] + all_motif_positions
    col_names = [all_cols[0]] + [all_cols[p] for p in all_motif_positions]

    print(f"Loading TSV ({len(all_motif_positions)} motif columns) ...")
    df = pd.read_csv(
        tsv_path,
        sep="\t",
        skiprows=1,
        header=None,
        usecols=use_cols,
        names=col_names,
        low_memory=False,
    )
    peak_id_col = col_names[0]
    df[peak_id_col] = df[peak_id_col].astype(str)
    n_peaks = len(df)
    print(f"Loaded {n_peaks:,} peaks")

    # Build peak-TF pairs, collapsing length variants
    records = []
    skipped_noisy = []

    for idx in sorted(index_to_col_positions, key=int):
        tf_name = index_to_tf[idx]
        positions = index_to_col_positions[idx]
        variant_col_names = [all_cols[p] for p in positions]

        # Hit = any length variant non-empty
        has_hit = pd.Series(False, index=df.index)
        for vcol in variant_col_names:
            if vcol in df.columns:
                has_hit = has_hit | (
                    df[vcol].notna() & (df[vcol].astype(str).str.strip() != "")
                )

        coverage = has_hit.sum() / n_peaks

        if coverage > max_coverage:
            skipped_noisy.append((idx, tf_name, f"{coverage:.1%}"))
            continue

        hit_peaks = df.loc[has_hit, peak_id_col].values
        for peak_id in hit_peaks:
            records.append({"peak_id": peak_id, "tf_name": tf_name})

    if skipped_noisy:
        print(f"\nDropped {len(skipped_noisy)} high-coverage (noisy) motifs "
              f"[threshold: >{max_coverage:.0%}]:")
        for idx, tf, cov in skipped_noisy:
            print(f"  motif {idx:>3} | {tf:<20} | {cov} of peaks")

    out = pd.DataFrame(records).drop_duplicates()
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--homer-dir",
                        default="motif_scanning/homer_motif_results")
    parser.add_argument("--output",
                        default="homer_outputs/peak_motifs.csv")
    parser.add_argument("--max-coverage", type=float, default=0.60,
                        help="Drop motifs hitting >this fraction of peaks (default: 0.60)")
    args = parser.parse_args()

    known_results_path  = os.path.join(args.homer_dir, "knownResults.txt")
    homer_results_dir   = os.path.join(args.homer_dir, "homerResults")
    annotation_tsv_path = os.path.join(args.homer_dir, "peak_motif_annotation.tsv")

    # Build index → TF name map
    known_rank_to_tf = parse_known_results(known_results_path)
    index_to_tf = build_index_to_tf(homer_results_dir, known_rank_to_tf)

    print("\nFinal index → TF name mapping:")
    for idx in sorted(index_to_tf, key=int)[:30]:
        print(f"  {idx:>3} → {index_to_tf[idx]}")
    if len(index_to_tf) > 30:
        print(f"  ... ({len(index_to_tf) - 30} more)")

    # Parse annotations
    out_df = parse_peak_annotation(annotation_tsv_path, index_to_tf, args.max_coverage)

    # Save
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    out_df.to_csv(args.output, index=False)

    print(f"\n{'='*50}")
    print(f"Done!")
    print(f"  {len(out_df):,} peak-TF pairs")
    print(f"  {out_df['tf_name'].nunique()} unique TF names")
    print(f"  {out_df['peak_id'].nunique():,} peaks with ≥1 motif hit")
    print(f"  Saved to {args.output}")

    print(f"\nTop TFs by peak count:")
    print(out_df["tf_name"].value_counts().head(20).to_string())

    # Sanity check: flag consensus-looking names
    consensus_re = re.compile(r"^\d+-[ACGTNWSRYKMBDHV]+$", re.IGNORECASE)
    bad = [t for t in out_df["tf_name"].unique() if consensus_re.match(t)]
    if bad:
        print(f"\nWARNING: {len(bad)} names still look like consensus sequences: {bad}")
    else:
        print(f"\nAll TF names look like gene symbols ✓")

    # Estimate how many will survive resolve_tf_gene_indices()
    likely_matchable = [
        t for t in out_df["tf_name"].unique()
        if re.match(r"^[A-Za-z][A-Za-z0-9]{1,9}$", t)
    ]
    print(f"~{len(likely_matchable)}/{out_df['tf_name'].nunique()} TFs have standard "
          f"gene symbol format (will likely match adata.var_names)")


if __name__ == "__main__":
    main()
