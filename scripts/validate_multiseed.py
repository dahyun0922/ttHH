#!/usr/bin/env python3
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import ast
import csv
import hashlib
import json
import gc
from pathlib import Path

import numpy as np
import awkward as ak
import uproot
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, accuracy_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SOURCE = Path("scripts/train_allpairs.py")
OUT = Path("results/dnn_multiseed")
OUT.mkdir(parents=True, exist_ok=True)

SAMPLES = ["ttHH", "HH", "ttbar"]
SEEDS = [42, 123, 2026, 777, 31415]
TEST_SEED = 271828

BASIC = [
    "nSelectedJet", "nBJets", "HT", "ST",
    "PuppiMET_pt", "leading_jet_pt",
    "subleading_jet_pt", "leading_lepton_pt"
]

STATS = [
    "min_mass", "mean_mass", "max_mass",
    "min_deltaR", "mean_deltaR", "max_deltaR"
]

BPAIR = ["bb_" + s for s in STATS]
JPAIR = ["jj_" + s for s in STATS]
ALL = BASIC + BPAIR + JPAIR

OBJECT = [
    "Jet_pt", "Jet_eta", "Jet_phi", "Jet_mass",
    "Jet_btagUParTAK4B",
    "Muon_pt", "Muon_eta", "Muon_phi",
    "Muon_mediumId", "Muon_pfRelIso04_all",
    "Electron_pt", "Electron_eta",
    "Electron_phi", "Electron_cutBased"
]

MODELS = {
    "A_basic": BASIC,
    "C_all_pairs": ALL
}

QUOTAS = {
    "train": 80000,
    "val": 20000,
    "test": 20000
}

with open("results/dnn_baseline/file_splits.json") as f:
    splits = json.load(f)

# Import only physics/data-selection function definitions.
# Do not execute the original training program.
tree = ast.parse(SOURCE.read_text())
names = {
    "delta_r",
    "selected_jets",
    "pair_stats",
    "sample_one"
}
nodes = [
    n for n in tree.body
    if isinstance(n, ast.FunctionDef) and n.name in names
]

if {n.name for n in nodes} != names:
    raise RuntimeError("Missing function in train_allpairs.py")

env = {
    "np": np,
    "ak": ak,
    "uproot": uproot,
    "splits": splits,
    "BASE": BASIC,
    "ALL": ALL,
    "OBJECT": OBJECT
}

exec(compile(
    ast.Module(body=nodes, type_ignores=[]),
    str(SOURCE), "exec"
), env)

sample_one = env["sample_one"]


def make_split(split, seed):
    env["rng"] = np.random.default_rng(seed)
    xx, yy = [], []

    for label, sample in enumerate(SAMPLES):
        x = sample_one(sample, split, QUOTAS[split])
        xx.append(x)
        yy.append(
            np.full(len(x), label, dtype=np.int32)
        )
        print(
            f"{split} / {sample}: {len(x)}",
            flush=True
        )

    x = np.concatenate(xx).astype(np.float32)
    y = np.concatenate(yy)

    order = env["rng"].permutation(len(y))
    return x[order], y[order]


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


# Fixed test data, prepared once and reused unchanged.
print("=== Preparing FIXED test set ===", flush=True)
Xtest, Ytest = make_split("test", TEST_SEED)

test_hash = hashlib.sha256(
    Xtest.tobytes() + Ytest.tobytes()
).hexdigest()

print("Fixed test SHA256:", test_hash, flush=True)

rows = []

for seed in SEEDS:
    print(f"\n========== SEED {seed} ==========", flush=True)

    # Different sampled train/validation events for each seed.
    Xtrain, Ytrain = make_split("train", seed)
    Xval, Yval = make_split("val", seed + 100000)

    scores = {}

    for name, features in MODELS.items():
        tf.keras.backend.clear_session()
        tf.keras.utils.set_random_seed(seed)

        idx = [ALL.index(f) for f in features]

        scaler = StandardScaler()
        train = scaler.fit_transform(
            Xtrain[:, idx]
        ).astype(np.float32)
        val = scaler.transform(
            Xval[:, idx]
        ).astype(np.float32)
        test = scaler.transform(
            Xtest[:, idx]
        ).astype(np.float32)

        model = build_model(len(features))

        history = model.fit(
            train, Ytrain,
            validation_data=(val, Yval),
            epochs=40,
            batch_size=1024,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(
                    monitor="val_loss",
                    patience=6,
                    restore_best_weights=True
                )
            ],
            verbose=0
        )

        probabilities = model.predict(
            test, batch_size=2048, verbose=0
        )
        predictions = probabilities.argmax(axis=1)

        auc = float(roc_auc_score(
            (Ytest == 0).astype(int),
            probabilities[:, 0]
        ))
        accuracy = float(accuracy_score(
            Ytest, predictions
        ))
        recall = float(np.mean(
            predictions[Ytest == 0] == 0
        ))

        scores[name] = {
            "auc": auc,
            "accuracy": accuracy,
            "recall": recall,
            "epochs": len(history.history["loss"])
        }

        print(
            f"{name}: AUC={auc:.5f}, "
            f"Accuracy={accuracy:.5f}, "
            f"Recall={recall:.5f}, "
            f"Epochs={scores[name]['epochs']}",
            flush=True
        )

        # Save separately; never overwrite previous studies.
        model.save(OUT / f"{name}_seed{seed}.keras")
        np.savez(
            OUT / f"{name}_seed{seed}_scaler.npz",
            features=np.array(features),
            mean=scaler.mean_,
            scale=scaler.scale_
        )

        del model, train, val, test, probabilities
        gc.collect()

    a = scores["A_basic"]
    c = scores["C_all_pairs"]

    row = {
        "seed": seed,
        "basic_auc": a["auc"],
        "allpairs_auc": c["auc"],
        "delta_auc": c["auc"] - a["auc"],
        "basic_accuracy": a["accuracy"],
        "allpairs_accuracy": c["accuracy"],
        "basic_recall": a["recall"],
        "allpairs_recall": c["recall"]
    }
    rows.append(row)

    # Save progress after each completed seed.
    with open(OUT / "results.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row))
        writer.writeheader()
        writer.writerows(rows)

    print(
        f"SEED {seed} Delta AUC = "
        f"{row['delta_auc']:+.5f}",
        flush=True
    )

    del Xtrain, Xval, Ytrain, Yval
    gc.collect()


deltas = np.array([r["delta_auc"] for r in rows])

summary = {
    "seeds": SEEDS,
    "fixed_test_seed": TEST_SEED,
    "fixed_test_sha256": test_hash,
    "test_events_per_class": QUOTAS["test"],
    "mean_delta_auc": float(np.mean(deltas)),
    "std_delta_auc": float(np.std(deltas, ddof=1)),
    "positive_delta_count": int(np.sum(deltas > 0)),
    "results": rows,
    "note": (
        "Unweighted balanced MC; fixed test sample; "
        "train/validation sampling and initialization varied. "
        "Standard deviation is across runs, not a physics "
        "significance or independent-test confidence interval."
    )
}

(OUT / "summary.json").write_text(
    json.dumps(summary, indent=2)
)

print("\n========== FINAL MULTI-SEED COMPARISON ==========")
for r in rows:
    print(
        f"Seed {r['seed']:5d}: "
        f"Basic={r['basic_auc']:.5f}, "
        f"All-pairs={r['allpairs_auc']:.5f}, "
        f"Delta={r['delta_auc']:+.5f}"
    )

print(
    f"Mean Delta AUC = {np.mean(deltas):+.5f} "
    f"+/- {np.std(deltas, ddof=1):.5f} (run-to-run SD)"
)
print(
    f"Positive improvements: {np.sum(deltas > 0)}/{len(deltas)}"
)

fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(
    SEEDS, [r["basic_auc"] for r in rows],
    "o-", label="Basic 8"
)
ax.plot(
    SEEDS, [r["allpairs_auc"] for r in rows],
    "s-", label="All-pairs 20"
)
ax.set_xlabel("Training seed")
ax.set_ylabel("ttHH ROC-AUC")
ax.legend()
fig.tight_layout()
fig.savefig(OUT / "multiseed_auc.png", dpi=180)
plt.close(fig)

print("SUCCESS: Multi-seed validation completed", flush=True)
