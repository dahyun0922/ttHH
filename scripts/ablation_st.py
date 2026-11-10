#!/usr/bin/env python3
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import csv
import json
from pathlib import Path

import numpy as np
import uproot
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, accuracy_score, classification_report
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = Path("results/dnn_baseline")
OUT = Path("results/dnn_ablation")
OUT.mkdir(parents=True, exist_ok=True)

CLASSES = ["ttHH", "HH", "ttbar"]
FEATURES = [
    "nSelectedJet", "nBJets", "HT", "ST",
    "PuppiMET_pt", "leading_jet_pt",
    "subleading_jet_pt", "leading_lepton_pt"
]

MODELS = {
    "with_ST": FEATURES,
    "without_ST": [f for f in FEATURES if f != "ST"]
}

SEED = 42
QUOTAS = {"train": 80000, "val": 20000, "test": 20000}

# Exactly the same ROOT file partition as baseline.
with open(BASE / "file_splits.json") as f:
    splits = json.load(f)

def load_events(files, quota, rng):
    xs = []
    remaining = quota

    for filename in files:
        if remaining == 0:
            break

        with uproot.open(f"{filename}:Events") as tree:
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
        x = x[valid]

        if len(x) == 0:
            continue

        count = min(remaining, 5000, len(x))
        indices = rng.choice(len(x), count, replace=False)

        xs.append(x[indices])
        remaining -= count

    if remaining:
        raise RuntimeError(
            f"Not enough events: {remaining} missing"
        )

    return np.concatenate(xs)


# Create ONE dataset shared by both models.
rng = np.random.default_rng(SEED)
X, Y = {}, {}

for split, quota in QUOTAS.items():
    parts_x, parts_y = [], []

    for label, sample in enumerate(CLASSES):
        x = load_events(splits[sample][split], quota, rng)
        parts_x.append(x)
        parts_y.append(
            np.full(len(x), label, dtype=np.int32)
        )
        print(
            f"Loaded {split}/{sample}: {len(x)}",
            flush=True
        )

    x = np.concatenate(parts_x)
    y = np.concatenate(parts_y)

    order = rng.permutation(len(y))
    X[split] = x[order]
    Y[split] = y[order]

print("Shared datasets prepared.", flush=True)


def build_model(input_dim):
    model = tf.keras.Sequential([
        tf.keras.Input(shape=(input_dim,)),
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
    return model


results = {}

for model_name, selected_features in MODELS.items():
    print(f"\n=== Training {model_name} ===", flush=True)

    # Same initialization seed for a controlled comparison.
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(SEED)

    columns = [FEATURES.index(f) for f in selected_features]

    scaler = StandardScaler()
    xtrain = scaler.fit_transform(
        X["train"][:, columns]
    ).astype(np.float32)
    xval = scaler.transform(
        X["val"][:, columns]
    ).astype(np.float32)
    xtest = scaler.transform(
        X["test"][:, columns]
    ).astype(np.float32)

    model = build_model(len(columns))

    history = model.fit(
        xtrain, Y["train"],
        validation_data=(xval, Y["val"]),
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

    prob = model.predict(
        xtest, batch_size=2048, verbose=0
    )
    pred = prob.argmax(axis=1)
    truth = Y["test"]

    metrics = {
        "ttHH_auc": float(roc_auc_score(
            (truth == 0).astype(int), prob[:, 0]
        )),
        "accuracy": float(accuracy_score(truth, pred)),
        "class_auc": {
            sample: float(roc_auc_score(
                (truth == i).astype(int), prob[:, i]
            ))
            for i, sample in enumerate(CLASSES)
        },
        "classification_report": classification_report(
            truth, pred,
            labels=[0, 1, 2],
            target_names=CLASSES,
            output_dict=True,
            zero_division=0
        ),
        "features": selected_features,
        "best_val_loss": float(min(
            history.history["val_loss"]
        ))
    }

    results[model_name] = metrics

    model.save(OUT / f"{model_name}.keras")
    np.savez(
        OUT / f"{model_name}_scaler.npz",
        mean=scaler.mean_,
        scale=scaler.scale_,
        features=np.array(selected_features)
    )

    print(
        f"{model_name}: ttHH AUC={metrics['ttHH_auc']:.5f}, "
        f"accuracy={metrics['accuracy']:.5f}",
        flush=True
    )

a = results["with_ST"]
b = results["without_ST"]

print("\n========== FINAL COMPARISON ==========")
print(f"With ST    : AUC={a['ttHH_auc']:.5f} "
      f"Accuracy={a['accuracy']:.5f}")
print(f"Without ST : AUC={b['ttHH_auc']:.5f} "
      f"Accuracy={b['accuracy']:.5f}")
print(
    "AUC difference (with - without): "
    f"{a['ttHH_auc'] - b['ttHH_auc']:+.5f}"
)
print(
    "Accuracy difference: "
    f"{a['accuracy'] - b['accuracy']:+.5f}"
)

with open(OUT / "comparison.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["model", "ttHH_auc", "accuracy"])
    for name, m in results.items():
        writer.writerow([
            name, m["ttHH_auc"], m["accuracy"]
        ])

(OUT / "comparison.json").write_text(
    json.dumps({
        "seed": SEED,
        "quotas": QUOTAS,
        "models": results,
        "note": (
            "Same sampled MC events for both models. "
            "Unweighted, class-balanced classification."
        )
    }, indent=2)
)

# AUC comparison plot.
names = ["With ST", "Without ST"]
values = [a["ttHH_auc"], b["ttHH_auc"]]

fig, ax = plt.subplots(figsize=(6, 4))
bars = ax.bar(names, values)
ax.set_ylim(0, 1)
ax.set_ylabel("ttHH ROC-AUC")
ax.set_title("ST Ablation Study")

for bar, value in zip(bars, values):
    ax.text(
        bar.get_x() + bar.get_width()/2,
        value + 0.015,
        f"{value:.4f}",
        ha="center"
    )

fig.tight_layout()
fig.savefig(OUT / "comparison.png", dpi=180)
plt.close(fig)

print("\nSUCCESS: ST ablation completed.", flush=True)
