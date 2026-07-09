# Sea Animal Classification

**23-class image classifier** using an EfficientNetB7 + ViT-B16 dual-backbone ensemble, optimized with Particle Swarm Optimization (PSO).  
Trained on ~12,000 images. Deployed with a Streamlit web dashboard featuring real-time inference and GradCAM visualization.

> Test accuracy: **93.0%** (PSO Ensemble)

---

## Results

| Model | Test Accuracy |
|---|---|
| EfficientNetB7 | 89.08% |
| ViT-B16 | 92.58% |
| **PSO Ensemble** | **93.00%** |

Ensemble weights found by PSO: `EfficientNet=0.382, ViT=0.618`

---

## Architecture

```
Input Image (256×256 RGB)
        │
   ┌────┴───────────────────────────┐
   │                                │
   ▼                                ▼
EfficientNetB7                  ViT-B16
(CNN — local textures)    (Transformer — global context)
   │                                │
   ▼                                ▼
 Softmax (23)               Softmax (23)
   │                                │
   └──────────┬─────────────────────┘
              ▼
     PSO weight search
  w_eff × p_eff + w_vit × p_vit
              ▼
       Final prediction
```

**EfficientNetB7** — Keras 3 on PyTorch backend
- Pretrained ImageNet weights, 66M parameters
- Custom head: Dense(256) → BN → Swish → Dropout(0.4) → Dense(128) → BN → Swish → Dropout(0.3) → Dense(23)
- Focal Loss (γ=2, α=0.25) to handle class imbalance
- Two-stage training: head-only (lr=1e-4), then top-50 layer fine-tune (lr=1e-5)
- Mixed precision (float16) via Keras

**ViT-B16** — Pure PyTorch + AMP
- Pretrained ImageNet-1K weights, 86M parameters  
- Position embeddings bicubic-interpolated from 14×14 → 16×16 for 256px input
- Custom head: LayerNorm → Dropout(0.4) → Linear(768→512) → GELU → Dropout(0.2) → Linear(512→23)
- CrossEntropyLoss with label smoothing ε=0.1
- MixUp (α=0.2), gradient accumulation (effective batch=32), gradient clipping (max=1.0)
- Two-stage training with differential learning rates

**PSO Ensemble**
- Searches for scalar weights [w_eff, w_vit] that maximize validation accuracy
- 30 particles, 60 iterations, pure numpy (~2 seconds, no GPU needed)

---

## Classes (23)

Clams · Corals · Crabs · Dolphin · Eel · Fish · Jelly Fish · Lobster · Nudibranchs · Octopus · Otter · Penguin · Puffers · Sea Rays · Sea Urchins · Seahorse · Seal · Sharks · Shrimp · Squid · Starfish · Turtle_Tortoise · Whale

---

## Project Structure

```
Sea_Animal_Classification/
├── sea_animals_main.ipynb        # Main training notebook
├── streamlit_app.py              # Web dashboard
├── src/
│   ├── config.py                 # All hyperparameters and paths
│   ├── dataset.py                # Data loading, splitting, DataLoaders
│   ├── efficientnet_model.py     # EfficientNetB7 build/train/load
│   ├── vit_model.py              # ViT-B16 build/train/load
│   ├── ensemble.py               # PSO optimizer
│   └── plots.py                  # Training curves, GradCAM, ELA, confusion matrix
│
├── dataset/                      # Images organized by class folder
├── checkpoints/                  # Saved model weights (auto-created)
├── ensemble_pso_config.json      # PSO weights + accuracy (written after training)
├── requirements.txt
└── README.md
```

---

## Setup

### Requirements

- Python 3.10
- CUDA 12.1
- GPU with ≥6 GB VRAM (tested on: RTX 4060 8 GB)

### Installation

```bash
git clone <repo-url>
cd Sea_Animal_Classification

python -m venv .venv-gpu
.venv-gpu\Scripts\activate   # Windows

# PyTorch with CUDA first
pip install torch==2.5.1+cu121 torchvision==0.20.1+cu121 \
    --index-url https://download.pytorch.org/whl/cu121

# Rest of dependencies
pip install -r requirements.txt
```

Verify CUDA is available:

```bash
python -c "import torch; print(torch.cuda.get_device_name(0))"
```

---

## Training

Open and run `sea_animals_main.ipynb` from top to bottom.

| Section | Description | Approx. time |
|---|---|---|
| 1 — Environment setup | CUDA check, VRAM info | instant |
| 2 — EDA | Class distribution, sample grid | < 1 min |
| 3 — ELA | Data quality inspection | < 1 min |
| 4 — Data split | Stratified train/val/test | < 2 min |
| 5–6 — EfficientNetB7 | Two-stage training | 60–90 min |
| 7–8 — ViT-B16 | Two-stage training | 50–70 min |
| 9–10 — PSO | Ensemble weight search | < 30 sec |
| 11 — Evaluation | Confusion matrix, classification report | < 1 min |
| 12–13 — Visualization | Prediction grid, GradCAM | < 2 min |

**Resuming interrupted training** — The notebook checks for existing checkpoints at startup:
```
checkpoints/efficientnet_sea.keras  →  EfficientNetB7 load, skip training
checkpoints/vit_b16_sea.pt          →  ViT-B16 load, skip training
```
Delete these files to retrain from scratch.

---

## Web Dashboard

```bash
.venv-gpu\Scripts\streamlit run streamlit_app.py
```

Upload any image to get:
- Top prediction + confidence score
- Class probability bar chart (top 5)
- GradCAM heatmap showing model attention regions
- Out-of-distribution warning if confidence < 40%

> The model is a closed-set classifier trained on 23 sea animal classes. Uploading an unrelated image will still produce a prediction — the confidence score indicates how certain the model is.

---

## Key Configuration

All training hyperparameters are in `src/config.py`:

```python
IMG_SIZE        = (256, 256)   # input resolution
BATCH_SIZE      = 16           # EfficientNetB7 (reduce to 8 if OOM)
VIT_BATCH       = 8            # ViT-B16 (reduce to 4 if OOM)
VIT_ACCUM       = 4            # gradient accumulation steps

EFF_EPOCHS_S1   = 20
EFF_EPOCHS_S2   = 15
VIT_EPOCHS_S1   = 12
VIT_EPOCHS_S2   = 15

LABEL_SMOOTHING = 0.10
MIXUP_ALPHA     = 0.20
DROPOUT_HEAD    = 0.40
PSO_PARTICLES   = 30
PSO_ITERS       = 60
```

---

## GradCAM

Gradient-weighted Class Activation Mapping (GradCAM) visualizes which spatial regions of the input the model used when making its prediction. Implemented via native PyTorch forward/backward hooks on EfficientNetB7's `top_activation` layer.

- **Red/yellow on the animal** → model is focusing on the correct region
- **Red/yellow on the background** → the model may be picking up spurious correlations (ocean color, reef texture)

---

## Troubleshooting

| Issue | Fix |
|---|---|
| `CUDA out of memory` (EfficientNet) | Set `BATCH_SIZE = 8` in `src/config.py` |
| `CUDA out of memory` (ViT) | Set `VIT_BATCH = 4` in `src/config.py` |
| DataLoader worker error on Windows | Set `NUM_WORKERS = 0` in `src/config.py` |
| `ModuleNotFoundError: src` | Run Jupyter from the project root directory |

---

## Dataset

- Source: [Kaggle — Sea Animals Image Dataset](https://www.kaggle.com)
- ~12,000 images across 23 classes
- Split: 64% train / 16% val / 20% test (stratified)

---


