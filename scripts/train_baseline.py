#!/usr/bin/env python3
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from pathlib import Path
import json
import numpy as np
import uproot
import tensorflow as tf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_curve, roc_auc_score
)

CLASSES = ["ttHH", "HH", "ttbar"]
FEATURES = [
    "nSelectedJet",
    "nBJets",
    "HT",
    "ST",
    "PuppiMET_pt",
    "leading_jet_pt",
    "subleading_jet_pt",
    "leading_lepton_pt"
]

OUT = Path("results/dnn_baseline")
OUT.mkdir(parents=True, exist_ok=True)

SEED = 42
rng = np.random.default_rng(SEED)
tf.keras.utils.set_random_seed(SEED)

# Split ROOT files, not individual events.
splits = {}
for label, sample in enumerate(CLASSES):
    files = sorted(
        Path(f"cache/features/{sample}").glob("[0-9][0-9][0-9][0-9].root")
    )
    if len(files) < 10:
        raise RuntimeError(f"Too few ROOT files: {sample}")

    files = [files[i] for i in rng.permutation(len(files))]
    ntrain = int(len(files) * 0.70)
    nval = int(len(files) * 0.15)

    splits[sample] = {
        "train": files[:ntrain],
        "val": files[ntrain:ntrain+nval],
        "test": files[ntrain+nval:]
    }

(OUT / "file_splits.json").write_text(json.dumps(
    {s: {k: [str(p) for p in v]
         for k, v in parts.items()}
     for s, parts in splits.items()},
    indent=2
))

def sample_events(files, quota, per_file=5000):
    """Bounded random-window sampling from each ROOT file."""
    xs, weights = [], []
    remaining = quota

    for path in files:
        if remaining <= 0:
            break

        with uproot.open(f"{path}:Events") as tree:
            n = tree.num_entries
            if n == 0:
                continue

            width = min(n, 20000)
            start = int(rng.integers(0, n-width+1))
            data = tree.arrays(
                FEATURES + ["genWeight"],
                entry_start=start,
                entry_stop=start+width,
                library="np"
            )

        x = np.column_stack(
            [data[f] for f in FEATURES]
        ).astype(np.float32)

        w = np.asarray(data["genWeight"], dtype=np.float64)

        valid = np.isfinite(x).all(axis=1) & np.isfinite(w)
        x, w = x[valid], w[valid]

        if len(x) == 0:
            continue

        take = min(remaining, per_file, len(x))
        indices = rng.choice(len(x), take, replace=False)

        xs.append(x[indices])
        weights.append(w[indices])
        remaining -= take

    if remaining > 0:
        raise RuntimeError(
            f"Insufficient events: missing {remaining}"
        )

    return np.concatenate(xs), np.concatenate(weights)


X, Y, W = {}, {}, {}

for split in ("train", "val", "test"):
    quota = 80000 if split == "train" else 20000

    xx, yy, ww = [], [], []

    for label, sample in enumerate(CLASSES):
        features, weights = sample_events(
            splits[sample][split], quota
        )

        xx.append(features)
        yy.append(np.full(len(features), label))
        ww.append(weights)

        print(
            f"{split} {sample}: {len(features)} events, "
            f"negative weights: {np.mean(weights < 0):.2%}",
            flush=True
        )

    x = np.concatenate(xx)
    y = np.concatenate(yy)
    w = np.concatenate(ww)

    order = rng.permutation(len(y))
    X[split], Y[split], W[split] = (
        x[order], y[order], w[order]
    )

# Fit normalization ONLY on training data.
scaler = StandardScaler()
X["train"] = scaler.fit_transform(X["train"])

for split in ("val", "test"):
    X[split] = scaler.transform(X[split])

np.savez(
    OUT / "scaler.npz",
    mean=scaler.mean_,
    scale=scaler.scale_,
    features=np.array(FEATURES)
)

# TensorFlow DNN
model = tf.keras.Sequential([
    tf.keras.Input(shape=(8,)),
    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dense(32, activation="relu"),
    tf.keras.layers.Dense(3, activation="softmax")
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

history = model.fit(
    X["train"], Y["train"],
    validation_data=(X["val"], Y["val"]),
    epochs=40,
    batch_size=1024,
    callbacks=[
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=6,
            restore_best_weights=True
        )
    ],
    verbose=2
)

model.save(OUT / "model.keras")

prob = model.predict(X["test"], verbose=0)
pred = np.argmax(prob, axis=1)
truth = Y["test"]

report = classification_report(
    truth, pred,
    labels=[0, 1, 2],
    target_names=CLASSES,
    digits=4,
    output_dict=True
)
print(classification_report(
    truth, pred,
    labels=[0, 1, 2],
    target_names=CLASSES,
    digits=4
))

# Confusion matrix
cm = confusion_matrix(
    truth, pred,
    labels=[0, 1, 2],
    normalize="true"
)

fig, ax = plt.subplots(figsize=(6, 5))
im = ax.imshow(cm, vmin=0, vmax=1)
ax.set_xticks(range(3), CLASSES)
ax.set_yticks(range(3), CLASSES)
ax.set_xlabel("Predicted")
ax.set_ylabel("True")

for i in range(3):
    for j in range(3):
        ax.text(j, i, f"{cm[i,j]:.3f}",
                ha="center", va="center")

fig.colorbar(im, ax=ax)
fig.tight_layout()
fig.savefig(OUT / "confusion_matrix.png", dpi=150)
plt.close(fig)

# ROC curves (one-vs-rest)
auc = {}
fig, ax = plt.subplots(figsize=(6, 5))

for i, sample in enumerate(CLASSES):
    binary_truth = (truth == i).astype(int)
    fpr, tpr, _ = roc_curve(
        binary_truth, prob[:, i]
    )
    score = roc_auc_score(binary_truth, prob[:, i])
    auc[sample] = float(score)
    ax.plot(fpr, tpr, label=f"{sample}: {score:.4f}")

ax.plot([0, 1], [0, 1], "--")
ax.set_xlabel("False positive rate")
ax.set_ylabel("True positive rate")
ax.legend()
fig.tight_layout()
fig.savefig(OUT / "roc.png", dpi=150)
plt.close(fig)

# Training history
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(history.history["loss"], label="Train")
ax.plot(history.history["val_loss"], label="Validation")
ax.set_xlabel("Epoch")
ax.set_ylabel("Cross entropy")
ax.legend()
fig.tight_layout()
fig.savefig(OUT / "loss.png", dpi=150)
plt.close(fig)

metrics = {
    "features": FEATURES,
    "classes": CLASSES,
    "seed": SEED,
    "train_per_class": 80000,
    "validation_per_class": 20000,
    "test_per_class": 20000,
    "auc": auc,
    "classification_report": report,
    "note": (
        "Unweighted, class-balanced MC baseline. "
        "Not a physical pp collision significance."
    )
}

(OUT / "metrics.json").write_text(
    json.dumps(metrics, indent=2)
)

print("SUCCESS: DNN training completed", flush=True)
print("Results:", OUT, flush=True)
