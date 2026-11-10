#!/usr/bin/env python3
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import ast
import csv
import json
from pathlib import Path

import numpy as np
import awkward as ak
import uproot
import tensorflow as tf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

SOURCE = Path("scripts/train_allpairs.py")
BASE = Path("results/dnn_baseline")
MODEL_DIR = Path("results/dnn_allpairs")
OUT = Path("results/dnn_efficiency")
OUT.mkdir(parents=True, exist_ok=True)

SAMPLES = ["ttHH", "HH", "ttbar"]
N_PER_CLASS = 20000
SEED = 20261110
TARGETS = [0.90, 0.80, 0.70, 0.50]

BASIC = [
    "nSelectedJet", "nBJets", "HT", "ST",
    "PuppiMET_pt", "leading_jet_pt",
    "subleading_jet_pt", "leading_lepton_pt"
]
STATS = [
    "min_mass", "mean_mass", "max_mass",
    "min_deltaR", "mean_deltaR", "max_deltaR"
]
FEATURES = (
    BASIC
    + ["bb_" + s for s in STATS]
    + ["jj_" + s for s in STATS]
)

OBJECT = [
    "Jet_pt", "Jet_eta", "Jet_phi", "Jet_mass",
    "Jet_btagUParTAK4B",
    "Muon_pt", "Muon_eta", "Muon_phi",
    "Muon_mediumId", "Muon_pfRelIso04_all",
    "Electron_pt", "Electron_eta",
    "Electron_phi", "Electron_cutBased"
]

with open(BASE / "file_splits.json") as f:
    splits = json.load(f)

# Load the previously tested physics functions only.
# The training code in train_allpairs.py is NOT run.
tree = ast.parse(SOURCE.read_text())
required = {"delta_r", "selected_jets", "pair_stats"}
nodes = [
    node for node in tree.body
    if isinstance(node, ast.FunctionDef)
    and node.name in required
]

if {node.name for node in nodes} != required:
    raise RuntimeError("Missing physics functions")

scope = {"np": np, "ak": ak}
exec(compile(
    ast.Module(body=nodes, type_ignores=[]),
    str(SOURCE), "exec"
), scope)

selected_jets = scope["selected_jets"]
pair_stats = scope["pair_stats"]
rng = np.random.default_rng(SEED)


def load_sample(sample):
    chunks = []
    remaining = N_PER_CLASS

    for feature_path in splits[sample]["test"]:
        if remaining <= 0:
            break

        object_path = feature_path.replace(
            "cache/features/", "cache/objects/", 1
        )

        with uproot.open(f"{feature_path}:Events") as ft, \
             uproot.open(f"{object_path}:Events") as ot:

            if ft.num_entries != ot.num_entries:
                raise RuntimeError(
                    f"Event count mismatch: {feature_path}"
                )

            n = ft.num_entries
            if n == 0:
                continue

            width = min(n, 20000)
            start = int(rng.integers(0, n-width+1))

            feature_branches = BASIC + [
                "nNonBJets", "genWeight"
            ]

            f = ft.arrays(
                feature_branches,
                entry_start=start,
                entry_stop=start+width,
                library="np"
            )
            o = ot.arrays(
                OBJECT,
                entry_start=start,
                entry_stop=start+width,
                library="ak"
            )

            # Check event alignment when identifiers exist.
            ids = [
                k for k in ("run", "luminosityBlock", "event")
                if k in ft and k in ot
            ]
            for key in ids:
                fe = ft[key].array(
                    entry_start=start,
                    entry_stop=start+width,
                    library="np"
                )
                oe = ot[key].array(
                    entry_start=start,
                    entry_stop=start+width,
                    library="np"
                )
                if not np.array_equal(fe, oe):
                    raise RuntimeError(
                        f"{key} mismatch: {feature_path}"
                    )

        basic = np.column_stack([
            f[k] for k in BASIC
        ]).astype(np.float32)

        valid = (
            np.isfinite(basic).all(axis=1)
            & np.isfinite(f["genWeight"])
        )
        candidates = np.flatnonzero(valid)
        if len(candidates) == 0:
            continue

        take_n = min(remaining, 5000, len(candidates))
        indices = rng.choice(
            candidates, take_n, replace=False
        )

        jets, bjets, nonbjets = selected_jets(o[indices])

        actual = np.column_stack([
            ak.to_numpy(ak.num(jets)),
            ak.to_numpy(ak.num(bjets)),
            ak.to_numpy(ak.num(nonbjets))
        ])
        expected = np.column_stack([
            f["nSelectedJet"][indices],
            f["nBJets"][indices],
            f["nNonBJets"][indices]
        ])

        if not np.array_equal(actual, expected):
            bad = np.flatnonzero(
                np.any(actual != expected, axis=1)
            )[0]
            raise RuntimeError(
                f"Jet selection mismatch: {feature_path}, "
                f"entry {start + int(indices[bad])}"
            )

        x = np.column_stack([
            basic[indices],
            pair_stats(bjets),
            pair_stats(nonbjets)
        ]).astype(np.float32)

        if not np.isfinite(x).all():
            raise RuntimeError(
                f"Non-finite input: {feature_path}"
            )

        chunks.append(x)
        remaining -= take_n

    if remaining:
        raise RuntimeError(
            f"{sample}: missing {remaining} test events"
        )

    return np.concatenate(chunks)


# Read independent held-out test files.
X, Y = [], []
for label, sample in enumerate(SAMPLES):
    x = load_sample(sample)
    X.append(x)
    Y.append(np.full(len(x), label, dtype=np.int32))
    print(f"Loaded {sample}: {len(x)} events", flush=True)

X = np.concatenate(X)
Y = np.concatenate(Y)

# Use the scaler from the trained 20-feature model.
scaler = np.load(
    MODEL_DIR / "C_all_pairs_scaler.npz"
)

if list(scaler["features"]) != FEATURES:
    raise RuntimeError("Model feature ordering mismatch")

X_scaled = (
    (X.astype(np.float64) - scaler["mean"])
    / scaler["scale"]
).astype(np.float32)

model = tf.keras.models.load_model(
    MODEL_DIR / "C_all_pairs.keras"
)

scores = model.predict(
    X_scaled, batch_size=2048, verbose=0
)[:, 0]

if not np.isfinite(scores).all():
    raise RuntimeError("Non-finite DNN scores")

auc = roc_auc_score((Y == 0).astype(int), scores)
print(f"\nTest ttHH one-vs-rest AUC: {auc:.5f}")

signal_scores = scores[Y == 0]

rows = []

for target in TARGETS:
    # Determine threshold from signal sample alone.
    threshold = float(np.quantile(
        signal_scores, 1.0 - target
    ))

    pass_mask = scores > threshold

    efficiencies = {
        sample: float(np.mean(pass_mask[Y == i]))
        for i, sample in enumerate(SAMPLES)
    }

    rows.append({
        "target_signal_eff": target,
        "score_threshold": threshold,
        "ttHH_eff": efficiencies["ttHH"],
        "HH_eff": efficiencies["HH"],
        "ttbar_eff": efficiencies["ttbar"]
    })


print("\n=== FINAL EFFICIENCY TABLE ===")
print("Target | Score cut | ttHH pass | HH pass | ttbar pass")
for r in rows:
    print(
        f"{r['target_signal_eff']:5.0%}  | "
        f"{r['score_threshold']:.6f} | "
        f"{r['ttHH_eff']:.4%} | "
        f"{r['HH_eff']:.4%} | "
        f"{r['ttbar_eff']:.4%}"
    )

with open(OUT / "efficiency_table.csv", "w", newline="") as f:
    writer = csv.DictWriter(
        f, fieldnames=list(rows[0].keys())
    )
    writer.writeheader()
    writer.writerows(rows)

(OUT / "summary.json").write_text(
    json.dumps({
        "model": "C_all_pairs",
        "test_events_per_class": N_PER_CLASS,
        "sampling_seed": SEED,
        "ttHH_auc": float(auc),
        "thresholds": rows,
        "note": (
            "Unweighted efficiencies on a class-balanced "
            "sample from held-out test files. "
            "Not full-MC signed-weight efficiencies."
        )
    }, indent=2)
)

# Plot score distributions, normalized per sample.
fig, ax = plt.subplots(figsize=(8, 5))
bins = np.linspace(0, 1, 51)

for i, sample in enumerate(SAMPLES):
    ax.hist(
        scores[Y == i],
        bins=bins,
        density=True,
        histtype="step",
        linewidth=1.7,
        label=sample
    )

ax.set_xlabel("DNN ttHH score")
ax.set_ylabel("Normalized density")
ax.set_title("All-pairs DNN: test MC scores")
ax.legend()
fig.tight_layout()
fig.savefig(OUT / "score_distributions.png", dpi=180)
plt.close(fig)

print("\nSUCCESS: Efficiency evaluation completed", flush=True)
