#!/usr/bin/env python3
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import json
from pathlib import Path

import numpy as np
import awkward as ak
import uproot
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, accuracy_score, classification_report
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path("results/dnn_baseline")
OUT = Path("results/dnn_pair_ablation")
OUT.mkdir(parents=True, exist_ok=True)

SAMPLES = ["ttHH", "HH", "ttbar"]

BASE = [
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
# Original pair statistics, followed by existence flags.
MASS = [
    f for f in BPAIR + JPAIR if "mass" in f
]
DR = [
    f for f in BPAIR + JPAIR if "deltaR" in f
]
FLAGS = ["bb_has_pair", "jj_has_pair"]

ALL = BASE + BPAIR + JPAIR + FLAGS

CONFIG = {
    "A_basic": BASE,
    "B_flags": BASE + FLAGS,
    "C_mass": BASE + MASS,
    "D_deltaR": BASE + DR,
    "E_allpairs": BASE + BPAIR + JPAIR
}

OBJECT = [
    "Jet_pt", "Jet_eta", "Jet_phi", "Jet_mass",
    "Jet_btagUParTAK4B",
    "Muon_pt", "Muon_eta", "Muon_phi",
    "Muon_mediumId", "Muon_pfRelIso04_all",
    "Electron_pt", "Electron_eta",
    "Electron_phi", "Electron_cutBased"
]

QUOTAS = {
    "train": 80000,
    "val": 20000,
    "test": 20000
}
SEED = 42

with open(ROOT / "file_splits.json") as f:
    splits = json.load(f)

rng = np.random.default_rng(SEED)


def delta_r(eta1, phi1, eta2, phi2):
    dphi = np.arctan2(
        np.sin(phi1 - phi2),
        np.cos(phi1 - phi2)
    )
    return np.sqrt((eta1 - eta2)**2 + dphi**2)


def selected_jets(o):
    # Verify against the current C++ Selection.h.
    mm = (
        (o["Muon_pt"] > 15)
        & (abs(o["Muon_eta"]) < 2.4)
        & o["Muon_mediumId"]
        & (o["Muon_pfRelIso04_all"] < 0.25)
    )

    em = (
        (o["Electron_pt"] > 15)
        & (abs(o["Electron_eta"]) < 2.5)
        & (o["Electron_cutBased"] >= 3)
    )

    le = ak.concatenate([
        o["Muon_eta"][mm],
        o["Electron_eta"][em]
    ], axis=1)

    lp = ak.concatenate([
        o["Muon_phi"][mm],
        o["Electron_phi"][em]
    ], axis=1)

    jp = o["Jet_pt"]
    je = o["Jet_eta"]
    jf = o["Jet_phi"]

    overlap = ak.any(
        delta_r(
            je[:, :, None], jf[:, :, None],
            le[:, None, :], lp[:, None, :]
        ) < 0.4,
        axis=2
    )

    keep = (
        (jp > 30)
        & (abs(je) < 2.4)
        & (~overlap)
    )

    jets = ak.zip({
        "pt": jp[keep],
        "eta": je[keep],
        "phi": jf[keep],
        "mass": o["Jet_mass"][keep],
        "btag": o["Jet_btagUParTAK4B"][keep]
    })

    bmask = jets.btag > 0.1272
    return jets, jets[bmask], jets[~bmask]


def pair_stats(jets):
    # No Higgs/W target mass is used anywhere.
    pairs = ak.combinations(
        jets, 2, fields=["a", "b"]
    )
    a, b = pairs.a, pairs.b

    dr = delta_r(a.eta, a.phi, b.eta, b.phi)

    px = (
        a.pt * np.cos(a.phi)
        + b.pt * np.cos(b.phi)
    )
    py = (
        a.pt * np.sin(a.phi)
        + b.pt * np.sin(b.phi)
    )
    pz = (
        a.pt * np.sinh(a.eta)
        + b.pt * np.sinh(b.eta)
    )

    ea = np.sqrt(
        (a.pt * np.cosh(a.eta))**2 + a.mass**2
    )
    eb = np.sqrt(
        (b.pt * np.cosh(b.eta))**2 + b.mass**2
    )

    mass2 = (
        (ea + eb)**2
        - px**2 - py**2 - pz**2
    )
    mass = np.sqrt(np.maximum(mass2, 0))

    values = []
    for v in (mass, dr):
        for fn in (ak.min, ak.mean, ak.max):
            result = fn(
                v, axis=1, mask_identity=True
            )
            result = ak.fill_none(result, -1)
            values.append(
                np.asarray(
                    ak.to_numpy(result),
                    dtype=np.float32
                )
            )

    return np.column_stack(values)


def sample_one(sample, split, quota):
    xparts = []
    remaining = quota

    for fn in splits[sample][split]:
        if remaining == 0:
            break

        opath = str(fn).replace(
            "cache/features/",
            "cache/objects/", 1
        )

        with uproot.open(f"{fn}:Events") as ft, \
             uproot.open(f"{opath}:Events") as ot:

            if ft.num_entries != ot.num_entries:
                raise RuntimeError(
                    f"Event count mismatch: {fn}"
                )

            n = ft.num_entries
            if n == 0:
                continue

            width = min(n, 20000)
            start = int(
                rng.integers(0, n-width+1)
            )

            branches = list(dict.fromkeys(
                BASE + ["nNonBJets", "genWeight"]
            ))

            f = ft.arrays(
                branches,
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
                fe = ft["event"].array(
                    entry_start=start,
                    entry_stop=start+width,
                    library="np"
                )
                oe = ot["event"].array(
                    entry_start=start,
                    entry_stop=start+width,
                    library="np"
                )
                if not np.array_equal(fe, oe):
                    raise RuntimeError(
                        f"Event order mismatch: {fn}"
                    )

        basic = np.column_stack([
            f[k] for k in BASE
        ]).astype(np.float32)

        good = (
            np.isfinite(basic).all(axis=1)
            & np.isfinite(f["genWeight"])
        )
        candidates = np.flatnonzero(good)
        if len(candidates) == 0:
            continue

        count = min(
            remaining, 5000, len(candidates)
        )
        take = rng.choice(
            candidates, count, replace=False
        )

        jets, bjets, njets = selected_jets(o[take])

        actual = np.column_stack([
            ak.to_numpy(ak.num(jets)),
            ak.to_numpy(ak.num(bjets)),
            ak.to_numpy(ak.num(njets))
        ])
        expected = np.column_stack([
            f["nSelectedJet"][take],
            f["nBJets"][take],
            f["nNonBJets"][take]
        ])

        if not np.array_equal(actual, expected):
            bad = np.where(
                np.any(actual != expected, axis=1)
            )[0]
            i = int(bad[0])
            raise RuntimeError(
                f"Selection mismatch in {fn}, "
                f"entry={start+int(take[i])}: "
                f"object={actual[i].tolist()}, "
                f"feature={expected[i].tolist()}. "
                "Check Selection.h and cleaning."
            )

        # Pair flags are derived from checked multiplicities.
        flags = np.column_stack([
            (actual[:, 1] >= 2).astype(np.float32),
            (actual[:, 2] >= 2).astype(np.float32)
        ])

        full = np.column_stack([
            basic[take],
            pair_stats(bjets),
            pair_stats(njets),
            flags
        ])

        if not np.isfinite(full).all():
            raise RuntimeError(
                f"Non-finite pair statistics: {fn}"
            )

        xparts.append(full.astype(np.float32))
        remaining -= count

    if remaining:
        raise RuntimeError(
            f"{sample}/{split}: "
            f"{remaining} events missing"
        )

    return np.concatenate(xparts)


# A, B, C share exactly the same sampled events.
X, Y = {}, {}

for split, quota in QUOTAS.items():
    parts, labels = [], []

    for i, sample in enumerate(SAMPLES):
        x = sample_one(sample, split, quota)
        parts.append(x)
        labels.append(
            np.full(len(x), i, dtype=np.int32)
        )
        print(
            f"{split}/{sample}: {len(x)}",
            flush=True
        )

    x = np.concatenate(parts)
    y = np.concatenate(labels)

    perm = rng.permutation(len(y))
    X[split] = x[perm]
    Y[split] = y[perm]


results = {}

for model_name, features in CONFIG.items():
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(SEED)

    idx = [ALL.index(v) for v in features]
    scaler = StandardScaler()

    train = scaler.fit_transform(
        X["train"][:, idx]
    ).astype(np.float32)
    val = scaler.transform(
        X["val"][:, idx]
    ).astype(np.float32)
    test = scaler.transform(
        X["test"][:, idx]
    ).astype(np.float32)

    model = tf.keras.Sequential([
        tf.keras.Input(shape=(len(idx),)),
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

    history = model.fit(
        train, Y["train"],
        validation_data=(val, Y["val"]),
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
        test, batch_size=2048, verbose=0
    )
    pred = prob.argmax(axis=1)

    auc = float(roc_auc_score(
        (Y["test"] == 0).astype(int),
        prob[:, 0]
    ))

    result = {
        "features": features,
        "ttHH_auc": auc,
        "accuracy": float(
            accuracy_score(Y["test"], pred)
        ),
        "ttHH_recall": float(
            np.mean(pred[Y["test"] == 0] == 0)
        ),
        "best_val_loss": float(
            min(history.history["val_loss"])
        ),
        "class_report": classification_report(
            Y["test"], pred,
            target_names=SAMPLES,
            output_dict=True
        )
    }
    results[model_name] = result

    model.save(OUT / f"{model_name}.keras")

    np.savez(
        OUT / f"{model_name}_scaler.npz",
        features=np.array(features),
        mean=scaler.mean_,
        scale=scaler.scale_
    )

    print(
        f"{model_name}: AUC={auc:.5f}, "
        f"Accuracy={result['accuracy']:.5f}, "
        f"Recall={result['ttHH_recall']:.5f}",
        flush=True
    )

(OUT / "comparison.json").write_text(
    json.dumps({
        "seed": SEED,
        "quotas": QUOTAS,
        "models": results,
        "note": (
            "Unweighted class-balanced MC; "
            "same sampled events for all models; "
            "one training seed."
        )
    }, indent=2)
)

print("\n=== FINAL COMPARISON ===")

for name, m in results.items():
    print(
        f"{name}: "
        f"ttHH AUC={m['ttHH_auc']:.5f}, "
        f"Accuracy={m['accuracy']:.5f}, "
        f"Recall={m['ttHH_recall']:.5f}"
    )

fig, ax = plt.subplots(figsize=(7, 4))
names = list(results)
values = [results[n]["ttHH_auc"] for n in names]
ax.bar(names, values)
ax.set_ylim(0, 1)
ax.set_ylabel("ttHH ROC-AUC")
fig.tight_layout()
fig.savefig(OUT / "comparison.png", dpi=160)
plt.close(fig)

print(
    "SUCCESS: pair-group ablation complete",
    flush=True
)
