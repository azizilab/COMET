"""comet.py — run with: python comet.py"""
import os
from config import CONFIG
from data.load import load_all_data
from training.training_loop import run_training


def main():
    os.makedirs(CONFIG["OUTPUT_DIR"], exist_ok=True)
    print("=" * 60); print("COSMOS — Loading data"); print("=" * 60)
    data = load_all_data(CONFIG)

    print("\n" + "=" * 60); print("COSMOS — Training"); print("=" * 60)
    model_config = {k: CONFIG[k] for k in ["D_ATOMS","ENCODER_HIDDEN","LD_HIDDEN","LD_DIM","VEL_HIDDEN"]}
    train_config = {k: CONFIG[k] for k in [
        "NUM_RUNS","N_EPOCHS","PHASE1_EPOCHS","BATCH_SIZE","CLIP_NORM","PLOT_EVERY",
        "USE_LOCAL_ATAC","N_CLUSTERS","NEIGHBOR_SPACE","LR_GRN","LR_VEL","ETA_MIN",
        "WEIGHTS_PHASE1","WEIGHTS_PHASE2","OUTPUT_DIR","MAX_DATALOADER_WORKERS",
        "OBS_PSEUDOTIME_KEY",
    ]}
    run_training(data, model_config, train_config)
    print("\nDone. Outputs in:", CONFIG["OUTPUT_DIR"])


if __name__ == "__main__":
    main()
