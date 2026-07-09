import os
import numpy as np
import keras
from keras import layers, Model
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from src.config import (
    IMG_SIZE, EFF_LR_HEAD, EFF_LR_FINETUNE,
    EFF_EPOCHS_S1, EFF_EPOCHS_S2, EFF_FINETUNE_LAYERS,
    LABEL_SMOOTHING, DROPOUT_HEAD,
    EFF_MODEL_PATH, EFF_CKPT_PATH, CHECKPOINT_DIR, LOG_DIR
)


def build_augmentation():
    h, w = IMG_SIZE
    return keras.Sequential([
        layers.Resizing(h, w),
        layers.RandomFlip("horizontal_and_vertical"),
        layers.RandomRotation(0.20),
        layers.RandomZoom(height_factor=0.15, width_factor=0.15),
        layers.RandomTranslation(height_factor=0.10, width_factor=0.10),
        layers.RandomContrast(0.15),
        layers.RandomBrightness(factor=0.15),
    ], name="augmentation")


def build_efficientnet(num_classes: int):
    import tensorflow as tf

    base = tf.keras.applications.EfficientNetB7(
        input_shape=(*IMG_SIZE, 3),
        include_top=False,
        weights='imagenet',
        pooling='max'
    )
    base.trainable = False

    augment = build_augmentation()

    inputs = layers.Input(shape=(*IMG_SIZE, 3))
    x = augment(inputs)
    x = base(x, training=False)
    x = layers.Dense(256, dtype='float32')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('swish')(x)
    x = layers.Dropout(DROPOUT_HEAD)(x)
    x = layers.Dense(128, dtype='float32')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('swish')(x)
    x = layers.Dropout(0.30)(x)
    outputs = layers.Dense(num_classes, activation='softmax', dtype='float32')(x)

    return Model(inputs, outputs), base


def focal_loss(gamma=2.0, alpha=0.25):
    def loss(y_true, y_pred):
        eps = keras.backend.epsilon()
        y_pred = keras.ops.clip(y_pred, eps, 1.0 - eps)
        ce = -y_true * keras.ops.log(y_pred)
        w  = alpha * keras.ops.power(1.0 - y_pred, gamma)
        return keras.ops.sum(w * ce, axis=-1)
    return loss


def get_class_weights(train_flow):
    from sklearn.utils.class_weight import compute_class_weight
    raw = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(train_flow.classes),
        y=train_flow.classes
    )
    return {int(k): float(v) for k, v in enumerate(raw)}


def train_efficientnet(train_flow, val_flow, num_classes):
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    model, base = build_efficientnet(num_classes)
    loss_fn     = focal_loss(gamma=2.0, alpha=0.25)
    class_w     = get_class_weights(train_flow)

    callbacks_s1 = [
        EarlyStopping(monitor='val_loss', patience=5,
                      restore_best_weights=True, verbose=1),
        ModelCheckpoint(EFF_CKPT_PATH, save_weights_only=True,
                        monitor='val_loss', save_best_only=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                          patience=3, min_lr=1e-6, verbose=1),
    ]

    model.compile(
        optimizer=Adam(EFF_LR_HEAD),
        loss=loss_fn,
        metrics=['accuracy',
                 keras.metrics.Precision(name='precision'),
                 keras.metrics.Recall(name='recall')]
    )

    print("EfficientNetB7 Stage 1: head training")
    hist1 = model.fit(
        train_flow,
        steps_per_epoch=len(train_flow),
        validation_data=val_flow,
        validation_steps=len(val_flow),
        epochs=EFF_EPOCHS_S1,
        class_weight=class_w,
        callbacks=callbacks_s1,
    )

    # Stage 2: fine-tune top layers
    base.trainable = True
    for layer in base.layers[:-EFF_FINETUNE_LAYERS]:
        layer.trainable = False

    callbacks_s2 = [
        EarlyStopping(monitor='val_loss', patience=5,
                      restore_best_weights=True, verbose=1),
        ModelCheckpoint(EFF_CKPT_PATH, save_weights_only=True,
                        monitor='val_loss', save_best_only=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                          patience=3, min_lr=1e-7, verbose=1),
    ]

    model.compile(
        optimizer=Adam(EFF_LR_FINETUNE),
        loss=loss_fn,
        metrics=['accuracy',
                 keras.metrics.Precision(name='precision'),
                 keras.metrics.Recall(name='recall')]
    )

    print("EfficientNetB7 Stage 2: fine-tuning")
    hist2 = model.fit(
        train_flow,
        steps_per_epoch=len(train_flow),
        validation_data=val_flow,
        validation_steps=len(val_flow),
        epochs=EFF_EPOCHS_S2,
        class_weight=class_w,
        callbacks=callbacks_s2,
    )

    model.save(EFF_MODEL_PATH)
    print(f"Saved: {EFF_MODEL_PATH}")

    return model, hist1, hist2


def load_efficientnet(num_classes: int):
    import tensorflow as tf
    if os.path.exists(EFF_MODEL_PATH):
        print(f"Loading EfficientNet from {EFF_MODEL_PATH}")
        model = keras.models.load_model(
            EFF_MODEL_PATH,
            custom_objects={'loss': focal_loss()}
        )
        return model
    raise FileNotFoundError(f"No saved model at {EFF_MODEL_PATH}")


def get_eff_probs(model, flow):
    """Run model.predict on a flow (shuffle=False), return softmax probs."""
    return model.predict(flow, verbose=0)
