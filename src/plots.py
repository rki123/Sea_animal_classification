import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix


def plot_curves(history_dict, stage_split=None):
    tr_acc  = history_dict['train_acc']
    vl_acc  = history_dict['val_acc']
    tr_loss = history_dict['train_loss']
    vl_loss = history_dict['val_loss']
    xs = range(len(tr_acc))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, tr, vl, title, ylabel in [
        (axes[0], tr_acc,  vl_acc,  'Accuracy', 'Accuracy'),
        (axes[1], tr_loss, vl_loss, 'Loss',     'Loss'),
    ]:
        ax.plot(xs, tr, 'b', label='Train')
        ax.plot(xs, vl, 'r', label='Val')
        if stage_split:
            ax.axvline(x=stage_split - 0.5, color='orange', ls='--', label='Fine-tune')
        ax.set_title(title); ax.set_xlabel('Epoch'); ax.set_ylabel(ylabel)
        ax.legend()
    plt.tight_layout(); plt.show()


def plot_confusion_matrix(y_true, y_pred, classes, figsize=(20, 14), text_size=9):
    cm = confusion_matrix(y_true, y_pred)
    norm_cm = cm.astype(float) / cm.sum(axis=1)[:, None]

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(norm_cm, annot=cm, fmt='d',
                cmap=sns.color_palette("crest", as_cmap=True),
                xticklabels=classes, yticklabels=classes,
                cbar=True, square=True, linewidths=0.5, ax=ax,
                annot_kws={"size": text_size * 0.85})
    ax.set_title("Confusion Matrix", fontsize=text_size + 4)
    ax.set_xlabel("Predicted", fontsize=text_size)
    ax.set_ylabel("True", fontsize=text_size)
    ax.set_xticklabels(classes, rotation=90, fontsize=text_size)
    ax.set_yticklabels(classes, fontsize=text_size)
    plt.tight_layout(); plt.show()


def plot_model_comparison(names, accuracies):
    colors = ['#4C72B0', '#DD8452', '#2ecc71'][:len(names)]
    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(names, accuracies, color=colors, width=0.45, edgecolor='white')
    for bar, a in zip(bars, accuracies):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.003,
                f'{a:.4f}', ha='center', va='bottom', fontsize=12, fontweight='bold')
    ax.set_ylim(0.80, 1.00)
    ax.set_title('Model Comparison - Test Accuracy')
    ax.set_ylabel('Accuracy')
    plt.tight_layout(); plt.show()


def plot_pso_convergence(history, naive_baseline=None):
    plt.figure(figsize=(10, 4))
    plt.plot(history, color='royalblue', lw=2, label='PSO best')
    if naive_baseline is not None:
        plt.axhline(naive_baseline, color='gray', ls='--', lw=1.2, label='50/50 baseline')
    plt.title('PSO Convergence'); plt.xlabel('Iteration'); plt.ylabel('Val Accuracy')
    plt.legend(); plt.grid(True, alpha=0.35)
    plt.tight_layout(); plt.show()


def plot_label_distribution(df):
    counts = df['Label'].value_counts()
    plt.figure(figsize=(20, 5))
    sns.barplot(x=counts.index, y=counts.values, palette='rocket')
    plt.title('Label Distribution'); plt.xlabel('Class'); plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.tight_layout(); plt.show()


def show_sample_grid(df, n=16, cols=4):
    indices = np.random.randint(0, len(df), n)
    rows = n // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols*3, rows*3),
                              subplot_kw={'xticks': [], 'yticks': []})
    for i, ax in enumerate(axes.flat):
        ax.imshow(plt.imread(df['Filepath'].iloc[indices[i]]))
        ax.set_title(df['Label'].iloc[indices[i]], fontsize=9)
    plt.tight_layout(); plt.show()


# ---------------------------------------------------------------------------
# ELA — Error Level Analysis
# What it is used for:
#   A data-quality inspection tool. When a JPEG is resaved at lower quality,
#   original regions compress consistently (low error = dark in ELA map).
#   Edited, manipulated, or watermarked regions re-compress differently
#   (high error = bright). Use ELA to:
#     1. Spot manipulated/watermarked images before training.
#     2. Find heavily re-saved images that will confuse the model.
#     3. Understand dataset composition.
# ELA does NOT improve training — it is a pre-training data audit step.
# ---------------------------------------------------------------------------

def _compute_ela(img_path, quality=90, amplify=15):
    """Re-compress in memory and return the amplified difference as RGB array."""
    from PIL import Image, ImageChops, ImageEnhance
    import io

    orig = Image.open(img_path).convert('RGB')
    buf  = io.BytesIO()
    orig.save(buf, format='JPEG', quality=quality)
    buf.seek(0)
    recomp = Image.open(buf).convert('RGB')

    ela      = ImageChops.difference(orig, recomp)
    extrema  = ela.getextrema()
    max_diff = max(ex[1] for ex in extrema) or 1
    ela      = ImageEnhance.Brightness(ela).enhance(255.0 / max_diff * amplify / 15)
    return np.array(ela)


def show_ela_grid(df, n=16, cols=4, quality=90, seed=42):
    """ELA map grid for n random images. Bright = high compression error."""
    rng     = np.random.default_rng(seed)
    indices = rng.integers(0, len(df), n)
    rows    = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3),
                              subplot_kw={'xticks': [], 'yticks': []})
    for i, ax in enumerate(axes.flat):
        if i >= n:
            ax.axis('off')
            continue
        idx = indices[i]
        try:
            ax.imshow(_compute_ela(df['Filepath'].iloc[idx], quality=quality))
        except Exception:
            ax.imshow(plt.imread(df['Filepath'].iloc[idx]))
        ax.set_title(df['Label'].iloc[idx], fontsize=8)

    plt.suptitle(
        f'ELA Grid  (JPEG quality={quality})  —  '
        'Bright regions = high compression error / possible edit',
        fontsize=10
    )
    plt.tight_layout(); plt.show()


def show_ela_vs_original(df, n=6, quality=90, seed=0):
    """Side-by-side original vs ELA map for n images."""
    rng     = np.random.default_rng(seed)
    indices = rng.integers(0, len(df), n)

    fig, axes = plt.subplots(n, 2, figsize=(8, n * 3),
                              subplot_kw={'xticks': [], 'yticks': []})
    if n == 1:
        axes = [axes]

    for row, idx in enumerate(indices):
        path  = df['Filepath'].iloc[idx]
        label = df['Label'].iloc[idx]
        try:
            axes[row][0].imshow(plt.imread(path))
            axes[row][0].set_title(f'Original — {label}', fontsize=9)
            axes[row][1].imshow(_compute_ela(path, quality=quality))
            axes[row][1].set_title(f'ELA  (q={quality})', fontsize=9)
        except Exception:
            pass

    plt.suptitle('Original vs ELA', fontsize=12)
    plt.tight_layout(); plt.show()


# ---------------------------------------------------------------------------
# Prediction grid — true vs predicted labels coloured green/red
# ---------------------------------------------------------------------------

def show_predictions_grid(test_df, pred_labels, n=40, cols=8):
    """
    Show n random test images with true and predicted labels.
    Green = correct, Red = wrong.
    """
    indices = np.random.randint(0, len(test_df), n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.5, rows * 3.5),
                              subplot_kw={'xticks': [], 'yticks': []})
    for i, ax in enumerate(axes.flat):
        if i >= n:
            ax.axis('off')
            continue
        idx      = indices[i]
        true_lbl = test_df['Label'].iloc[idx]
        pred_lbl = pred_labels[idx]
        ax.imshow(plt.imread(test_df['Filepath'].iloc[idx]))
        color = 'green' if true_lbl == pred_lbl else 'red'
        ax.set_title(f'T: {true_lbl}\nP: {pred_lbl}', color=color, fontsize=8)
    plt.suptitle('Predictions — Green: Correct  |  Red: Wrong', fontsize=13)
    plt.tight_layout(); plt.show()


# ---------------------------------------------------------------------------
# GradCAM — works with EfficientNetB7 (Keras 3 + PyTorch backend)
# ---------------------------------------------------------------------------

def _find_layer(model, name):
    """Recursively search for a layer by name inside nested models."""
    for layer in model.layers:
        if layer.name == name:
            return layer
        if hasattr(layer, 'layers'):
            found = _find_layer(layer, name)
            if found:
                return found
    return None


def _gradcam_heatmap(keras_model, img_array, last_conv_layer_name='top_activation',
                     pred_index=None):
    """
    Compute GradCAM heatmap for one image.
    img_array: numpy array shape (1, H, W, 3), already preprocessed.
    Returns: (heatmap np.ndarray H×W in [0,1], predicted_class_index)
    """
    import torch
    import keras

    last_conv = _find_layer(keras_model, last_conv_layer_name)
    if last_conv is None:
        raise ValueError(f"Layer '{last_conv_layer_name}' not found in model.")

    grad_model = keras.Model(
        inputs=keras_model.inputs,
        outputs=[last_conv.output, keras_model.output]
    )

    img_t = torch.tensor(img_array, dtype=torch.float32)

    with torch.enable_grad():
        conv_out, preds = grad_model(img_t, training=False)
        conv_out = conv_out.float()
        preds    = preds.float()
        conv_out.retain_grad()

        if pred_index is None:
            pred_index = int(preds[0].detach().argmax())

        preds[0, pred_index].backward()

    grads = conv_out.grad.detach()
    feats = conv_out.detach()

    # Handle channels-first (B,C,H,W) and channels-last (B,H,W,C)
    if feats.shape[1] <= feats.shape[-1]:
        # channels-last: (B, H, W, C)
        pooled  = grads.mean(dim=(0, 1, 2))          # (C,)
        heatmap = (feats[0] * pooled).sum(dim=-1)     # (H, W)
    else:
        # channels-first: (B, C, H, W)
        pooled  = grads.mean(dim=(0, 2, 3))                    # (C,)
        heatmap = (feats[0] * pooled[:, None, None]).sum(dim=0) # (H, W)

    heatmap = torch.relu(heatmap)
    mx = heatmap.max()
    if mx > 0:
        heatmap = heatmap / mx

    return heatmap.numpy(), pred_index


def _overlay_heatmap(img_path, heatmap, alpha=0.45):
    """Overlay jet-coloured heatmap on the original image."""
    import cv2
    from PIL import Image

    orig = np.array(Image.open(img_path).convert('RGB'))
    h, w = orig.shape[:2]

    hmap_resized  = cv2.resize(heatmap, (w, h))
    hmap_uint8    = np.uint8(255 * hmap_resized)
    jet_colors    = plt.colormaps['jet'](hmap_uint8)[:, :, :3]
    jet_rgb       = np.uint8(jet_colors * 255)
    superimposed  = np.uint8(jet_rgb * alpha + orig * (1 - alpha))
    return superimposed


def show_gradcam_grid(keras_model, test_df, pred_labels, classes,
                      n=15, cols=5,
                      last_conv_layer='top_activation'):
    """
    Show GradCAM overlays for n random test images.
    Falls back to original image if GradCAM computation fails.
    """
    import tensorflow as tf
    from src.config import IMG_SIZE

    preprocess = tf.keras.applications.efficientnet.preprocess_input
    indices    = np.random.randint(0, len(test_df), n)
    rows       = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4),
                              subplot_kw={'xticks': [], 'yticks': []})

    for i, ax in enumerate(axes.flat):
        if i >= n:
            ax.axis('off')
            continue

        idx      = indices[i]
        img_path = test_df['Filepath'].iloc[idx]
        true_lbl = test_df['Label'].iloc[idx]
        pred_lbl = pred_labels[idx]

        try:
            import keras as _keras
            img       = _keras.utils.load_img(img_path, target_size=IMG_SIZE)
            img_array = _keras.utils.img_to_array(img)
            img_array = preprocess(np.expand_dims(img_array, axis=0))

            heatmap, _ = _gradcam_heatmap(keras_model, img_array, last_conv_layer)
            display    = _overlay_heatmap(img_path, heatmap)
        except Exception:
            display = plt.imread(img_path)

        ax.imshow(display)
        color = 'green' if true_lbl == pred_lbl else 'red'
        ax.set_title(f'True: {true_lbl}\nPred: {pred_lbl}', color=color, fontsize=9)

    plt.suptitle('GradCAM — EfficientNetB7  (Green: Correct  |  Red: Wrong)',
                 fontsize=13)
    plt.tight_layout(); plt.show()
