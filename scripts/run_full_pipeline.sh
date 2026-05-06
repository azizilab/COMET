#!/bin/bash

set -e  # stop immediately if any step fails

echo "=============================="
echo "STARTING ATAC PIPELINE"
date
echo "=============================="

# ----------------------------
# STEP 1: R PIPELINE
# ----------------------------
echo "Running R pipeline..."
Rscript atac_pipeline.R > atac.log 2>&1

echo "R pipeline finished"
date

# ----------------------------
# STEP 2: PYTHON PIPELINE
# ----------------------------
echo "Running Python TF-gene pipeline..."
python tf_gene_matrix.py > tf.log 2>&1

echo "Python pipeline finished"
date

echo "=============================="
echo "ALL PIPELINES COMPLETE"
echo "=============================="
