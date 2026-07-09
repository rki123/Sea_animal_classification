import os
import copy
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision.models import vit_b_16, ViT_B_16_Weights

from src.config import (
    VIT_EPOCHS_S1, VIT_EPOCHS_S2, VIT_ACCUM,
    VIT_LR_HEAD, VIT_LR_ENCODER, VIT_LR_HEAD_S2,
    VIT_UNFREEZE_BLOCKS, LABEL_SMOOTHING, MIXUP_ALPHA,
    DROPOUT_HEAD, VIT_CKPT_PATH, CHECKPOINT_DIR, VIT_IMG_SIZE
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _interpolate_pos_embed(model, new_img_size=256, patch_size=16):
    """Bicubic-interpolate position embeddings from 224x224 to new_img_size."""
    old_pos = model.encoder.pos_embedding.data      # (1, 197, 768)
    cls_tok = old_pos[:, :1, :]                     # (1, 1, 768)
    patches = old_pos[:, 1:, :]                     # (1, 196, 768)

    old_n = int(patches.shape[1] ** 0.5)            # 14
    new_n = new_img_size // patch_size              # e.g. 16 for 256px

    if old_n == new_n:
        return

    patches = patches.reshape(1, old_n, old_n, -1).permute(0, 3, 1, 2).float()
    patches = F.interpolate(patches, size=(new_n, new_n), mode='bicubic', align_corners=False)
    patches = patches.permute(0, 2, 3, 1).reshape(1, new_n * new_n, -1)

    model.encoder.pos_embedding = nn.Parameter(torch.cat([cls_tok, patches], dim=1))
    print(f"ViT pos-embed interpolated: {old_n}x{old_n} -> {new_n}x{new_n}")


def build_vit(num_classes: int):
    model = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
    _interpolate_pos_embed(model, new_img_size=VIT_IMG_SIZE)
    model.image_size = VIT_IMG_SIZE

    for p in model.parameters():
        p.requires_grad = False

    in_f = model.heads.head.in_features
    model.heads = nn.Sequential(
        nn.LayerNorm(in_f),
        nn.Dropout(DROPOUT_HEAD),
        nn.Linear(in_f, 512),
        nn.GELU(),
        nn.Dropout(DROPOUT_HEAD * 0.5),
        nn.Linear(512, num_classes),
    )
    for p in model.heads.parameters():
        p.requires_grad = True

    return model.to(DEVICE)


def _mixup(x, y, alpha=MIXUP_ALPHA):
    lam = float(np.random.beta(alpha, alpha)) if alpha > 0 else 1.0
    idx = torch.randperm(x.size(0), device=x.device)
    return lam * x + (1 - lam) * x[idx], y, y[idx], lam


class _EarlyStopping:
    def __init__(self, patience=6):
        self.patience   = patience
        self.best_score = None
        self.counter    = 0
        self.best_state = None

    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None or score > self.best_score + 1e-4:
            self.best_score = score
            self.best_state = copy.deepcopy(model.state_dict())
            self.counter    = 0
        else:
            self.counter += 1
        return self.counter >= self.patience

    def restore(self, model):
        if self.best_state:
            model.load_state_dict(self.best_state)


def _make_criterion():
    return nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING).to(DEVICE)


def _train_epoch(model, loader, opt, scaler, criterion, accum=VIT_ACCUM):
    model.train()
    tot_loss = correct = total = 0
    opt.zero_grad()
    for step, (imgs, labels) in enumerate(loader):
        imgs   = imgs.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)
        with autocast():
            imgs_m, ya, yb, lam = _mixup(imgs, labels)
            logits = model(imgs_m)
            loss   = (lam * criterion(logits, ya) + (1 - lam) * criterion(logits, yb)) / accum

        scaler.scale(loss).backward()
        if (step + 1) % accum == 0 or (step + 1) == len(loader):
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            opt.zero_grad()

        tot_loss += loss.item() * accum
        correct  += (logits.argmax(1) == labels).sum().item()
        total    += labels.size(0)
    return tot_loss / len(loader), correct / total


@torch.no_grad()
def _eval_epoch(model, loader, criterion):
    model.eval()
    tot_loss = correct = total = 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(DEVICE, non_blocking=True), labels.to(DEVICE, non_blocking=True)
        with autocast():
            logits = model(imgs)
            loss   = criterion(logits, labels)
        tot_loss += loss.item()
        correct  += (logits.argmax(1) == labels).sum().item()
        total    += labels.size(0)
    return tot_loss / len(loader), correct / total


def train_vit(tr_dl, vl_dl, num_classes):
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    model     = build_vit(num_classes)
    criterion = _make_criterion()

    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

    # Stage 1: head only
    print("ViT-B16 Stage 1: head training")
    opt_s1    = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                            lr=VIT_LR_HEAD, weight_decay=0.01)
    sched_s1  = CosineAnnealingLR(opt_s1, T_max=VIT_EPOCHS_S1, eta_min=1e-5)
    scaler_s1 = GradScaler()
    stopper   = _EarlyStopping(patience=4)

    for ep in range(VIT_EPOCHS_S1):
        t0 = time.time()
        tr_l, tr_a = _train_epoch(model, tr_dl, opt_s1, scaler_s1, criterion)
        vl_l, vl_a = _eval_epoch(model, vl_dl, criterion)
        sched_s1.step()
        history['train_loss'].append(tr_l); history['val_loss'].append(vl_l)
        history['train_acc'].append(tr_a);  history['val_acc'].append(vl_a)
        print(f"  S1 ep {ep+1:02d}  tr_loss={tr_l:.4f} tr_acc={tr_a:.4f} "
              f"vl_loss={vl_l:.4f} vl_acc={vl_a:.4f} [{time.time()-t0:.0f}s]")
        if stopper(vl_l, model):
            print("  Early stop S1")
            break

    stopper.restore(model)
    torch.cuda.empty_cache()

    # Stage 2: unfreeze last N encoder blocks
    print(f"ViT-B16 Stage 2: unfreezing last {VIT_UNFREEZE_BLOCKS} encoder blocks")
    for block in model.encoder.layers[-VIT_UNFREEZE_BLOCKS:]:
        for p in block.parameters():
            p.requires_grad = True
    for p in model.encoder.ln.parameters():
        p.requires_grad = True

    enc_params  = [p for n, p in model.named_parameters()
                   if p.requires_grad and 'heads' not in n]
    head_params = list(model.heads.parameters())
    opt_s2 = optim.AdamW([
        {'params': enc_params,  'lr': VIT_LR_ENCODER,  'weight_decay': 0.01},
        {'params': head_params, 'lr': VIT_LR_HEAD_S2, 'weight_decay': 0.005},
    ])
    sched_s2  = CosineAnnealingLR(opt_s2, T_max=VIT_EPOCHS_S2, eta_min=1e-7)
    scaler_s2 = GradScaler()
    stopper2  = _EarlyStopping(patience=5)

    for ep in range(VIT_EPOCHS_S2):
        t0 = time.time()
        tr_l, tr_a = _train_epoch(model, tr_dl, opt_s2, scaler_s2, criterion)
        vl_l, vl_a = _eval_epoch(model, vl_dl, criterion)
        sched_s2.step()
        history['train_loss'].append(tr_l); history['val_loss'].append(vl_l)
        history['train_acc'].append(tr_a);  history['val_acc'].append(vl_a)
        print(f"  S2 ep {ep+1:02d}  tr_loss={tr_l:.4f} tr_acc={tr_a:.4f} "
              f"vl_loss={vl_l:.4f} vl_acc={vl_a:.4f} [{time.time()-t0:.0f}s]")
        if stopper2(vl_l, model):
            print("  Early stop S2")
            break

    stopper2.restore(model)
    torch.cuda.empty_cache()

    torch.save({'model_state_dict': model.state_dict()}, VIT_CKPT_PATH)
    print(f"Saved: {VIT_CKPT_PATH}")
    return model, history


def load_vit(num_classes: int):
    if not os.path.exists(VIT_CKPT_PATH):
        raise FileNotFoundError(f"No checkpoint at {VIT_CKPT_PATH}")
    model = build_vit(num_classes)
    # unfreeze all to load correctly, then weights will reflect trained state
    for p in model.parameters():
        p.requires_grad = True
    ckpt = torch.load(VIT_CKPT_PATH, map_location=DEVICE, weights_only=True)
    model.load_state_dict(ckpt['model_state_dict'])
    model.to(DEVICE)
    print(f"Loaded ViT from {VIT_CKPT_PATH}")
    return model


@torch.no_grad()
def get_vit_probs(model, loader):
    model.eval()
    out = []
    for batch in loader:
        imgs = batch[0] if isinstance(batch, (list, tuple)) else batch
        imgs = imgs.to(DEVICE, non_blocking=True)
        with autocast():
            logits = model(imgs)
        out.append(torch.softmax(logits, 1).cpu().numpy())
    return np.vstack(out)
