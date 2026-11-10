#!/usr/bin/env python3
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import csv
import json
from pathlib import Path

import numpy as np
import uproot
import tensorflow as tf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, roc_curve,
    accuracy_score, classification_report,
    confusion_matrix
)

BASE = Path("results/dnn_baseline")
OUT = Path("results/dnn_reco_comparison")
OUT.mkdir(parents=True, exist_ok=True)

CLASSES = ["ttHH", "HH", "ttbar"]

BASIC = [
    "nSelectedJet",
    "nBJets",
    "HT",
    "ST",
    "PuppiMET_pt",
    "leading_jet_pt",
    "subleading_jet_pt",
    "leading_lepton_pt"
]

MODELS = {
    "A_basic": BASIC,
    "B_nonb": BASIC + [
        "nNonBJets"
    ],
    "C_mass": BASIC + [
        "nNonBJets",
        "m_bb",
        "m_W_had"
    ],
    "D_reco": BASIC + [
        "nNonBJets",
        "m_bb",
        "m_W_had",
        "pt_bb",
        "deltaR_bb_Hcand"
    ]
}

ALL_FEATURES = MODELS["D_reco"]
SEED = 42
QUOTAS = {
    "train": 80000,
    "val": 20000,
    "test": 20000
}

with open(BASE / "file_splits.json") as f:
    splits = json.load(f)


def read_sample(files, quota, rng):
    chunks = []
    remaining = quota

    for filename in files:
        if remaining <= 0:
            break

        with uproot.open(f"{filename}:Events") as tree:
            n = tree.num_entries
            if n == 0:
                continue

            width = min(n, 20000)
            start = int(rng.integers(0, n-width+1))

            data = tree.arrays(
                ALL_FEATURES + ["genWeight"],
                entry_start=start,
                entry_stop=start+width,
                library="np"
            )

        x = np.column_stack(
            [data[k] for k in ALL_FEATURES]
        ).astype(np.float32)

        w = np.asarray(
            data["genWeight"], dtype=np.float64
        )

        valid = (
            np.isfinite(x).all(axis=1)
            & np.isfinite(w)
        )
        x = x[valid]

        if len(x) == 0:
            continue

        take = min(remaining, 5000, len(x))
        idx = rng.choice(len(x), take, replace=False)
        chunks.append(x[idx])
        remaining -= take

    if remaining:
        raise RuntimeError(
            f"Not enough events: {remaining} missing"
        )

    return np.concatenate(chunks)


# Prepare ONE common dataset for all four models.
rng = np.random.default_rng(SEED)
X, Y = {}, {}

for split, quota in QUOTAS.items():
    xx, yy = [], []

    for label, sample in enumerate(CLASSES):
        x = read_sample(splits[sample][split], quota, rng)
        xx.append(x)
        yy.append(
            np.full(len(x), label, dtype=np.int32)
        )
        print(
            f"{split}/{sample}: {len(x)} events",
            flush=True
        )

    x = np.concatenate(xx)
    y = np.concatenate(yy)

    order = rng.permutation(len(y))
    X[split] = x[order]
    Y[split] = y[order]

print("Shared dataset prepared.", flush=True)


def build_model(n_features):
    model = tf.keras.Sequential([
        tf.keras.Input(shape=(n_features,)),
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
roc_data = {}

for name, selected in MODELS.items():
    print(f"\n===== Training {name} =====", flush=True)

    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(SEED)

    columns = [ALL_FEATURES.index(f) for f in selected]

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

    model = build_model(len(selected))

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
    truth = Y["test"]
    pred = np.argmax(prob, axis=1)

    aucs = {}
    for i, sample in enumerate(CLASSES):
        binary = (truth == i).astype(int)
        aucs[sample] = float(
            roc_auc_score(binary, prob[:, i])
        )

    fpr, tpr, _ = roc_curve(
        (truth == 0).astype(int), prob[:, 0]
    )
    roc_data[name] = (fpr, tpr)

    results[name] = {
        "features": selected,
        "ttHH_auc": aucs["ttHH"],
        "class_auc": aucs,
        "accuracy": float(accuracy_score(truth, pred)),
        "ttHH_recall": float(
            np.mean(pred[truth == 0] == 0)
        ),
        "best_val_loss": float(
            min(history.history["val_loss"])
        ),
        "classification_report": classification_report(
            truth, pred,
            labels=[0, 1, 2],
            target_names=CLASSES,
            output_dict=True,
            zero_division=0
        ),
        "confusion_matrix": confusion_matrix(
            truth, pred, labels=[0, 1, 2]
        ).tolist()
    }

    model.save(OUT / f"{name}.keras")

    np.savez(
        OUT / f"{name}_scaler.npz",
        mean=scaler.mean_,
        scale=scaler.scale_,
        features=np.array(selected)
    )

    print(
        f"{name}: "
        f"ttHH AUC={aucs['ttHH']:.5f}, "
        f"Accuracy={results[name]['accuracy']:.5f}, "
        f"ttHH Recall={results[name]['ttHH_recall']:.5f}",
        flush=True
    )


print("\n========== FINAL COMPARISON ==========")
previous_auc = None

for name, m in results.items():
    delta = (
        0.0 if previous_auc is None
        else m["ttHH_auc"] - previous_auc
    )
    print(
        f"{name:10s} | "
        f"AUC={m['ttHH_auc']:.5f} | "
        f"Accuracy={m['accuracy']:.5f} | "
        f"Recall={m['ttHH_recall']:.5f} | "
        f"Delta AUC={delta:+.5f}"
    )
    previous_auc = m["ttHH_auc"]

with open(OUT / "comparison.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "model", "n_features", "ttHH_auc",
        "accuracy", "ttHH_recall"
    ])
    for name, m in results.items():
        writer.writerow([
            name,
            len(m["features"]),
            m["ttHH_auc"],
            m["accuracy"],
            m["ttHH_recall"]
        ])

(OUT / "comparison.json").write_text(
    json.dumps({
        "seed": SEED,
        "quotas": QUOTAS,
        "models": results,
        "note": (
            "All models evaluated on identical sampled "
            "events. Unweighted and class-balanced MC. "
            "One training seed only."
        )
    }, indent=2)
)

# Comparison of ttHH ROC curves.
fig, ax = plt.subplots(figsize=(7, 6))
for name, (fpr, tpr) in roc_data.items():
    auc = results[name]["ttHH_auc"]
    ax.plot(fpr, tpr, label=f"{name} (AUC={auc:.4f})")

ax.plot([0, 1], [0, 1], "--", color="gray")
ax.set_xlabel("Background efficiency (FPR)")
ax.set_ylabel("ttHH efficiency (TPR)")
ax.set_title("ttHH ROC: Feature-set comparison")
ax.legend()
fig.tight_layout()
fig.savefig(OUT / "roc_comparison.png", dpi=180)
plt.close(fig)

# Bar chart for AUC.
names = list(results)
aucs = [results[n]["ttHH_auc"] for n in names]

fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.bar(names, aucs)
ax.set_ylim(0, 1)
ax.set_ylabel("ttHH ROC-AUC")
ax.set_title("Feature-set Ablation")

for bar, value in zip(bars, aucs):
    ax.text(
        bar.get_x()+bar.get_width()/2,
        value+0.015,
        f"{value:.4f}",
        ha="center"
    )

fig.tight_layout()
fig.savefig(OUT / "auc_comparison.png", dpi=180)
plt.close(fig)

print("\nSUCCESS: Four-model comparison completed.",
      flush=True)
