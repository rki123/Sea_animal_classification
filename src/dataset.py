import os
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from sklearn.model_selection import train_test_split

from src.config import (
    DATASET_DIR, IMG_SIZE, VIT_IMG_SIZE, BATCH_SIZE, VIT_BATCH,
    NUM_WORKERS, IMAGENET_MEAN, IMAGENET_STD
)

ALLOWED_EXT = {'.jpg', '.jpeg', '.png', '.bmp'}


def build_dataframe(dataset_dir: str) -> pd.DataFrame:
    image_dir = Path(dataset_dir)
    filepaths = []
    for ext in ['*.jpg', '*.JPG', '*.jpeg', '*.png', '*.PNG']:
        filepaths.extend(image_dir.glob(f'**/{ext}'))

    labels = [p.parent.name for p in filepaths]
    df = pd.DataFrame({'Filepath': [str(p) for p in filepaths], 'Label': labels})
    return df


def is_valid_image(path: str, min_hw: int = 32) -> bool:
    try:
        with Image.open(path) as img:
            img.verify()
        with Image.open(path) as img:
            w, h = img.size
            return w >= min_hw and h >= min_hw
    except Exception:
        return False


def clean_df(df: pd.DataFrame) -> pd.DataFrame:
    mask = []
    for _, row in df.iterrows():
        p = row['Filepath']
        ok = (Path(p).suffix.lower() in ALLOWED_EXT
              and Path(p).exists()
              and is_valid_image(p))
        mask.append(ok)
    clean = df[mask].reset_index(drop=True)
    print(f"Removed {len(df) - len(clean)} invalid images. Remaining: {len(clean)}")
    return clean


def make_splits(df: pd.DataFrame, test_size=0.2, val_size=0.2, seed=42):
    train_df, test_df = train_test_split(
        df, test_size=test_size, random_state=seed, stratify=df['Label']
    )
    train_df, val_df = train_test_split(
        train_df, test_size=val_size, random_state=seed, stratify=train_df['Label']
    )
    train_df = train_df.reset_index(drop=True)
    val_df   = val_df.reset_index(drop=True)
    test_df  = test_df.reset_index(drop=True)
    print(f"Split: train={len(train_df)}  val={len(val_df)}  test={len(test_df)}")
    return train_df, val_df, test_df


# --- PyTorch Dataset (used by ViT and inference) ---

class SeaAnimalDataset(Dataset):
    def __init__(self, df: pd.DataFrame, class_to_idx: dict,
                 transform=None, return_label=True):
        self.df           = df.reset_index(drop=True)
        self.class_to_idx = class_to_idx
        self.transform    = transform
        self.return_label = return_label

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        try:
            img = Image.open(row['Filepath']).convert('RGB')
        except Exception:
            img = Image.new('RGB', (224, 224))
        if self.transform:
            img = self.transform(img)
        if self.return_label:
            return img, self.class_to_idx[row['Label']]
        return img


def vit_transforms():
    sz = VIT_IMG_SIZE
    train_tf = T.Compose([
        T.Resize((sz + 32, sz + 32)),
        T.RandomResizedCrop(sz, scale=(0.65, 1.0)),
        T.RandomHorizontalFlip(0.5),
        T.RandomVerticalFlip(0.15),
        T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
        T.RandomRotation(20),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        T.RandomErasing(p=0.15, scale=(0.02, 0.12)),
    ])
    eval_tf = T.Compose([
        T.Resize((sz, sz)),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    return train_tf, eval_tf


def make_vit_loaders(train_df, val_df, test_df, class_to_idx):
    train_tf, eval_tf = vit_transforms()
    nw = NUM_WORKERS

    tr_ds = SeaAnimalDataset(train_df, class_to_idx, transform=train_tf)
    vl_ds = SeaAnimalDataset(val_df,   class_to_idx, transform=eval_tf)
    te_ds = SeaAnimalDataset(test_df,  class_to_idx, transform=eval_tf, return_label=False)

    tr_dl = DataLoader(tr_ds, batch_size=VIT_BATCH, shuffle=True,
                       num_workers=nw, pin_memory=True, drop_last=False,
                       persistent_workers=(nw > 0))
    vl_dl = DataLoader(vl_ds, batch_size=VIT_BATCH, shuffle=False,
                       num_workers=nw, pin_memory=True,
                       persistent_workers=(nw > 0))
    te_dl = DataLoader(te_ds, batch_size=VIT_BATCH, shuffle=False,
                       num_workers=0, pin_memory=True)
    return tr_dl, vl_dl, te_dl


# --- Keras/TF data generators (EfficientNet) ---

def make_keras_generators(train_df, val_df, test_df):
    import tensorflow as tf
    from tensorflow.keras.preprocessing.image import ImageDataGenerator

    preprocess = tf.keras.applications.efficientnet.preprocess_input

    train_gen = ImageDataGenerator(preprocessing_function=preprocess)
    val_gen   = ImageDataGenerator(preprocessing_function=preprocess)
    test_gen  = ImageDataGenerator(preprocessing_function=preprocess)

    common = dict(x_col='Filepath', y_col='Label', target_size=IMG_SIZE,
                  color_mode='rgb', class_mode='categorical', batch_size=BATCH_SIZE)

    train_flow = train_gen.flow_from_dataframe(
        dataframe=train_df, shuffle=True, seed=42, **common)
    val_flow = val_gen.flow_from_dataframe(
        dataframe=val_df, shuffle=False, **common)
    test_flow = test_gen.flow_from_dataframe(
        dataframe=test_df, shuffle=False, **common)

    return train_flow, val_flow, test_flow
