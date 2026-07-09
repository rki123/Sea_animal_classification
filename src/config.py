import os

# Must be set before importing keras
os.environ["KERAS_BACKEND"] = "torch"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# Paths
DATASET_DIR   = "dataset"
CHECKPOINT_DIR = "checkpoints"
LOG_DIR        = "training_logs"

# Image settings
# 256x256 gives measurable accuracy gains over 224x224 while staying safe on 8 GB VRAM.
# Reduce BATCH_SIZE to 8 if you see CUDA OOM errors during EfficientNetB7 training.
IMG_SIZE     = (256, 256)
VIT_IMG_SIZE = 256        # used for position-embedding interpolation in ViT
BATCH_SIZE   = 16
NUM_WORKERS  = 2      # set 0 if DataLoader errors on Windows

# EfficientNet training
EFF_LR_HEAD     = 1e-4
EFF_LR_FINETUNE = 1e-5
EFF_EPOCHS_S1   = 20
EFF_EPOCHS_S2   = 15
EFF_FINETUNE_LAYERS = 50

# ViT training
VIT_BATCH       = 8             # batch 8 at 256x256 is safe on 8 GB VRAM
VIT_ACCUM       = 4             # effective batch = 8 * 4 = 32
VIT_LR_HEAD     = 3e-4
VIT_LR_ENCODER  = 5e-6
VIT_LR_HEAD_S2  = 5e-5
VIT_EPOCHS_S1   = 12
VIT_EPOCHS_S2   = 15
VIT_UNFREEZE_BLOCKS = 4   # last N encoder blocks to unfreeze in stage 2

# Regularisation (shared)
LABEL_SMOOTHING = 0.10
MIXUP_ALPHA     = 0.20
DROPOUT_HEAD    = 0.40

# PSO
PSO_PARTICLES = 30
PSO_ITERS     = 60

# Saved model names
EFF_MODEL_PATH = "checkpoints/efficientnet_sea.keras"
EFF_CKPT_PATH  = "checkpoints/efficientnet_sea.weights.h5"
VIT_CKPT_PATH  = "checkpoints/vit_b16_sea.pt"
ENSEMBLE_CFG   = "ensemble_pso_config.json"

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]
