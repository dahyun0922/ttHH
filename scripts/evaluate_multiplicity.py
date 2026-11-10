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
from sklearn.metrics import roc_auc_score

SOURCE = Path("scripts/train_allpairs.py")
BASE_DIR = Path("results/dnn_baseline")
MODEL_DIR = Path("results/dnn_allpairs")
OUT = Path("results/dnn_multiplicity")
OUT.mkdir(parents=True, exist_ok=True)

SAMPLES = ["ttHH", "HH", "ttbar"]
N_PER_CLASS = 20000
SEED = 20261110

BASIC = [
    "nSelectedJet", "nBJets", "HT", "ST",
    "PuppiMET_pt", "leading_jet_pt",
    "subleading_jet_pt", "leading_lepton_pt"
]
STATS = [
    "min_mass", "mean_mass", "max_mass",
    "min_deltaR", "mean_deltaR", "max_deltaR"
]
BB = ["bb_" + x for x in STATS]
JJ = ["jj_" + x for x in STATS]
ALL = BASIC + BB + JJ

OBJECT = [
    "Jet_pt", "Jet_eta", "Jet_phi", "Jet_mass",
    "Jet_btagUParTAK4B",
    "Muon_pt", "Muon_eta", "Muon_phi",
    "Muon_mediumId", "Muon_pfRelIso04_all",
    "Electron_pt", "Electron_eta",
    "Electron_phi", "Electron_cutBased"
]

# Safely reuse the three tested physics functions.
# Do not execute the training code at module level.
source_text = SOURCE.read_text()
tree = ast.parse(source_text)
needed = ["delta_r", "selected_jets", "pair_stats"]

nodes = [
    node for node in tree.body
    if isinstance(node, ast.FunctionDef)
    and node.name in needed
]
if sorted(n.name for n in nodes) != sorted(needed):
    raise RuntimeError("Physics functions missing in train_allpairs.py")

namespace = {"np": np, "ak": ak}
exec(compile(
    ast.Module(body=nodes, type_ignores=[]),
    str(SOURCE), "exec"
), namespace)

selected_jets = namespace["selected_jets"]
pair_stats = namespace["pair_stats"]

with open(BASE_DIR / "file_splits.json") as f:
    splits = json.load(f)

rng = np.random.default_rng(SEED)

def load_test(sample):
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

            f = ft.arrays(
                BASIC + ["nNonBJets", "genWeight"],
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

            if "event" in ft and "event" in ot:
                a = ft["event"].array(
                    entry_start=start,
                    entry_stop=start+width,
                    library="np"
                )
                b = ot["event"].array(
                    entry_start=start,
                    entry_stop=start+width,
                    library="np"
                )
                if not np.array_equal(a, b):
                    raise RuntimeError(
                        f"Event order mismatch: {feature_path}"
                    )

        basic = np.column_stack(
            [f[k] for k in BASIC]
        ).astype(np.float32)

        valid = (
            np.isfinite(basic).all(axis=1)
            & np.isfinite(f["genWeight"])
        )
        candidates = np.flatnonzero(valid)
        if len(candidates) == 0:
            continue

        take_n = min(remaining, 5000, len(candidates))
        idx = rng.choice(
            candidates, take_n, replace=False
        )

        jets, bjets, nonbjets = selected_jets(o[idx])

        actual = np.column_stack([
            ak.to_numpy(ak.num(jets)),
            ak.to_numpy(ak.num(bjets)),
            ak.to_numpy(ak.num(nonbjets))
        ])
        expected = np.column_stack([
            f["nSelectedJet"][idx],
            f["nBJets"][idx],
            f["nNonBJets"][idx]
        ])

        if not np.array_equal(actual, expected):
            bad = np.flatnonzero(
                np.any(actual != expected, axis=1)
            )[0]
            raise RuntimeError(
                f"Selection mismatch: {feature_path}, "
                f"entry {start + int(idx[bad])}"
            )

        x = np.column_stack([
            basic[idx],
            pair_stats(bjets),
            pair_stats(nonbjets)
        ]).astype(np.float32)

        if not np.isfinite(x).all():
            raise RuntimeError(
                f"Non-finite features: {feature_path}"
            )

        chunks.append(x)
        remaining -= take_n

    if remaining > 0:
        raise RuntimeError(
            f"{sample}: missing {remaining} test events"
        )

    return np.concatenate(chunks)

xx, yy = [], []
for label, sample in enumerate(SAMPLES):
    x = load_test(sample)
    xx.append(x)
    yy.append(np.full(len(x), label, dtype=np.int32))
    print(f"Loaded {sample}: {len(x)} events", flush=True)

X = np.concatenate(xx)
Y = np.concatenate(yy)

def predict_model(name, columns):
    model = tf.keras.models.load_model(
        MODEL_DIR / f"{name}.keras"
    )
    scaler = np.load(MODEL_DIR / f"{name}_scaler.npz")

    stored_features = list(scaler["features"])
    wanted_features = [ALL[i] for i in columns]
    if stored_features != wanted_features:
        raise RuntimeError(
            f"Feature order mismatch for {name}"
        )

    transformed = (
        (X[:, columns].astype(np.float64) - scaler["mean"])
        / scaler["scale"]
    ).astype(np.float32)

    score = model.predict(
        transformed, batch_size=2048, verbose=0
    )[:, 0]
    return score

score_a = predict_model(
    "A_basic", list(range(8))
)
score_e = predict_model(
    "C_all_pairs", list(range(20))
)

NB = X[:, BASIC.index("nBJets")].astype(int)
NN = NB * 0 + (
    X[:, BASIC.index("nSelectedJet")].astype(int) - NB
)

def auc_for(mask, score, background=None):
    selected = mask.copy()
    if background is not None:
        selected &= ((Y == 0) | (Y == background))

    n_signal = int(np.sum(selected & (Y == 0)))
    n_background = int(np.sum(selected & (Y != 0)))

    if n_signal < 100 or n_background < 100:
        return n_signal, n_background, None

    target = (Y[selected] == 0).astype(int)
    auc = roc_auc_score(target, score[selected])
    return n_signal, n_background, float(auc)

rows = []

def compare(name, mask):
    for bgname, bg in [
        ("HH+ttbar", None),
        ("HH", 1),
        ("ttbar", 2)
    ]:
        ns, nb, auc_a = auc_for(mask, score_a, bg)
        _, _, auc_e = auc_for(mask, score_e, bg)

        if auc_a is None or auc_e is None:
            continue

        row = {
            "selection": name,
            "background": bgname,
            "n_ttHH": ns,
            "n_background": nb,
            "auc_basic": auc_a,
            "auc_allpairs": auc_e,
            "delta_auc": auc_e - auc_a
        }
        rows.append(row)

# Inclusive and threshold selections.
compare("All events", np.ones(len(Y), dtype=bool))
compare("nB>=2", NB >= 2)
compare("nB>=2,nNonB>=2", (NB >= 2) & (NN >= 2))
compare("nB>=3,nNonB>=2", (NB >= 3) & (NN >= 2))

# Exact multiplicity groups with sufficient statistics.
for nb in range(2, 6):
    for nn in range(2, 7):
        compare(
            f"nB={nb},nNonB={nn}",
            (NB == nb) & (NN == nn)
        )

output = OUT / "multiplicity_auc.csv"
with open(output, "w", newline="") as f:
    writer = csv.DictWriter(
        f, fieldnames=list(rows[0].keys())
    )
    writer.writeheader()
    writer.writerows(rows)

print("\n=== FINAL MULTIPLICITY COMPARISON ===")
for r in rows:
    print(
        f"{r['selection']:23s} "
        f"vs {r['background']:8s} | "
        f"S={r['n_ttHH']:5d} "
        f"B={r['n_background']:5d} | "
        f"A={r['auc_basic']:.4f} "
        f"E={r['auc_allpairs']:.4f} "
        f"Delta={r['delta_auc']:+.4f}"
    )

(OUT / "summary.json").write_text(
    json.dumps({
        "seed": SEED,
        "events_per_class": N_PER_CLASS,
        "rows": rows,
        "note": (
            "Unweighted MC. Both models evaluated on "
            "identical new samples from held-out test files. "
            "The exact test events may differ from prior runs."
        )
    }, indent=2)
)

print("\nSUCCESS: Multiplicity validation completed")
