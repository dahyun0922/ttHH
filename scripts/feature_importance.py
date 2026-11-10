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
from sklearn.metrics import roc_auc_score, accuracy_score

BASE = Path("results/dnn_baseline")
CLASSES = ["ttHH", "HH", "ttbar"]
FEATURES = [
    "nSelectedJet",
    "nBJets",
    "HT",
    "ST",
    "PuppiMET_pt",
    "leading_jet_pt",
    "subleading_jet_pt",
    "leading_lepton_pt",
]

SEED = 20261110
N_PER_CLASS = 20000
REPEATS = 5

rng = np.random.default_rng(SEED)

print("Loading trained model...", flush=True)
model = tf.keras.models.load_model(BASE / "model.keras")
scaler = np.load(BASE / "scaler.npz")
mean = scaler["mean"]
scale = scaler["scale"]

with open(BASE / "file_splits.json") as f:
    splits = json.load(f)

# Load only held-out test ROOT files.
def load_sample(sample):
    xs = []
    remaining = N_PER_CLASS

    for filename in splits[sample]["test"]:
        if remaining <= 0:
            break

        with uproot.open(f"{filename}:Events") as tree:
            n = tree.num_entries
            if n == 0:
                continue

            width = min(n, 20000)
            start = int(rng.integers(0, n - width + 1))

            data = tree.arrays(
                FEATURES,
                entry_start=start,
                entry_stop=start + width,
                library="np",
            )

        x = np.column_stack(
            [data[name] for name in FEATURES]
        ).astype(np.float32)

        x = x[np.isfinite(x).all(axis=1)]
        if len(x) == 0:
            continue

        take = min(remaining, 5000, len(x))
        indices = rng.choice(len(x), take, replace=False)
        xs.append(x[indices])
        remaining -= take

    if remaining:
        raise RuntimeError(
            f"{sample}: insufficient test events, missing {remaining}"
        )

    return np.concatenate(xs)


X_parts, y_parts = [], []

for label, sample in enumerate(CLASSES):
    x = load_sample(sample)
    X_parts.append(x)
    y_parts.append(np.full(len(x), label, dtype=np.int32))
    print(f"Loaded {sample}: {len(x)} events", flush=True)

X = np.concatenate(X_parts)
y = np.concatenate(y_parts)

# Use the scaler fitted on the original training set.
X = ((X - mean) / scale).astype(np.float32)

binary_truth = (y == 0).astype(np.int32)

def evaluate(x):
    scores = model.predict(x, batch_size=2048, verbose=0)
    return {
        "ttHH_auc": roc_auc_score(binary_truth, scores[:, 0]),
        "accuracy": accuracy_score(y, scores.argmax(axis=1)),
    }

baseline = evaluate(X)
print("\nBaseline on this held-out sample:", baseline, flush=True)

results = []

for j, name in enumerate(FEATURES):
    drops = []
    acc_drops = []

    for rep in range(REPEATS):
        shuffled = X.copy()
        shuffled[:, j] = rng.permutation(shuffled[:, j])

        modified = evaluate(shuffled)
        drops.append(baseline["ttHH_auc"] - modified["ttHH_auc"])
        acc_drops.append(baseline["accuracy"] - modified["accuracy"])

    result = {
        "feature": name,
        "auc_drop_mean": float(np.mean(drops)),
        "auc_drop_std": float(np.std(drops)),
        "accuracy_drop_mean": float(np.mean(acc_drops)),
    }
    results.append(result)

    print(
        f"{name:<22} "
        f"AUC drop = {result['auc_drop_mean']:.5f} "
        f"+/- {result['auc_drop_std']:.5f}",
        flush=True
    )

results.sort(key=lambda r: r["auc_drop_mean"], reverse=True)

output_csv = BASE / "permutation_importance.csv"
with open(output_csv, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(results[0]))
    writer.writeheader()
    writer.writerows(results)

# Plot
names = [r["feature"] for r in results][::-1]
values = [r["auc_drop_mean"] for r in results][::-1]
errors = [r["auc_drop_std"] for r in results][::-1]

fig, ax = plt.subplots(figsize=(9, 5))
ax.barh(names, values, xerr=errors, capsize=3)
ax.axvline(0, linestyle="--", linewidth=1)
ax.set_xlabel("Decrease in ttHH ROC-AUC after permutation")
ax.set_title("Permutation Feature Importance")
fig.tight_layout()
fig.savefig(BASE / "feature_importance.png", dpi=180)
plt.close(fig)

(BASE / "importance_summary.json").write_text(
    json.dumps({
        "baseline": baseline,
        "events_per_class": N_PER_CLASS,
        "repeats": REPEATS,
        "results": results,
        "note": "Unweighted held-out MC; not physical significance."
    }, indent=2)
)

print("\n=== Feature importance ranking ===")
for rank, r in enumerate(results, start=1):
    print(f"{rank}. {r['feature']}: {r['auc_drop_mean']:.5f}")

print("\nSUCCESS: Feature importance analysis completed")
