"""
Trains MobileNetV2 + LSTM for activity classification.
Input: .npy files in ml/datasets/activity_processed/{split}/{class}/
Output: ml/models/activity_mobilenet_lstm/activity_model.keras
        ml/models/activity_mobilenet_lstm/class_names.json
Run: python ml/training/train_activity_lstm.py
"""
import os, json, numpy as np
from pathlib import Path
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks

# ── Config ────────────────────────────────────────────────────────────────────
SEQ_LEN     = 30
IMG_SIZE    = (224, 224)
BATCH_SIZE  = 8
EPOCHS      = 20
DATA_ROOT   = Path("ml/datasets/activity_processed")
MODEL_OUT   = Path("ml/models/activity_mobilenet_lstm")
MODEL_OUT.mkdir(parents=True, exist_ok=True)

# ── Data loading ──────────────────────────────────────────────────────────────
def load_split(split: str):
    X, y, class_names = [], [], []
    split_dir = DATA_ROOT / split
    if not split_dir.exists():
        return np.array([]), np.array([]), []

    classes = sorted([d.name for d in split_dir.iterdir() if d.is_dir()])
    for label, cls in enumerate(classes):
        if split == "train":
            class_names.append(cls)
        for npy in (split_dir / cls).glob("*.npy"):
            seq = np.load(str(npy)).astype("float32") / 255.0
            X.append(seq)
            y.append(label)

    return np.array(X), np.array(y), classes

print("Loading data...")
X_train, y_train, class_names = load_split("train")
X_val,   y_val,   _           = load_split("val")

print(f"Train: {X_train.shape}, Val: {X_val.shape}")
print(f"Classes: {class_names}")

# Save class names
with open(MODEL_OUT / "class_names.json", "w") as f:
    json.dump(class_names, f)

# ── Model: MobileNetV2 (frame encoder) + LSTM (sequence) ─────────────────────
base = tf.keras.applications.MobileNetV2(
    input_shape=(*IMG_SIZE, 3), include_top=False, pooling="avg", weights="imagenet"
)
base.trainable = False   # freeze during initial training

frame_input = layers.Input(shape=(SEQ_LEN, *IMG_SIZE, 3))
# TimeDistributed applies MobileNetV2 to each frame
x = layers.TimeDistributed(base)(frame_input)
x = layers.LSTM(128, return_sequences=False)(x)
x = layers.Dropout(0.4)(x)
x = layers.Dense(64, activation="relu")(x)
x = layers.Dense(len(class_names), activation="softmax")(x)

model = models.Model(frame_input, x)
model.compile(
    optimizer=optimizers.Adam(1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)
model.summary()

# ── Callbacks ─────────────────────────────────────────────────────────────────
cb = [
    callbacks.ModelCheckpoint(
        str(MODEL_OUT / "activity_model.keras"),
        save_best_only=True, monitor="val_accuracy", verbose=1
    ),
    callbacks.EarlyStopping(patience=5, restore_best_weights=True),
    callbacks.ReduceLROnPlateau(patience=3, factor=0.5),
]

# ── Train ─────────────────────────────────────────────────────────────────────
model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    callbacks=cb,
)

# Fine-tune: unfreeze top layers of MobileNetV2
base.trainable = True
for layer in base.layers[:-20]:
    layer.trainable = False

model.compile(optimizer=optimizers.Adam(1e-5), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
model.fit(X_train, y_train, validation_data=(X_val, y_val), batch_size=BATCH_SIZE, epochs=5, callbacks=cb)

print(f"\n✅ Model saved → {MODEL_OUT / 'activity_model.keras'}")
print(f"✅ Classes saved → {MODEL_OUT / 'class_names.json'}")