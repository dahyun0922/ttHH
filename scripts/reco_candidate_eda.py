#!/usr/bin/env python3
import json
from pathlib import Path

import numpy as np
import uproot
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = Path("results/dnn_baseline")
OUT = Path("results/reco_eda")
OUT.mkdir(parents=True, exist_ok=True)

SAMPLES = ["ttHH", "HH", "ttbar"]
N_TARGET = 40000
SEED = 42

BRANCHES = [
    "m_bb", "m_W_had",
    "has_bb_candidate", "has_jj_candidate",
    "nBJets", "nNonBJets"
]

rng = np.random.default_rng(SEED)

with open(BASE / "file_splits.json") as f:
    splits = json.load(f)

data_by_sample = {}

for sample in SAMPLES:
    files = splits[sample]["test"]
    chunks = []
    remaining = N_TARGET

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
                BRANCHES,
                entry_start=start,
                entry_stop=start+width,
                library="np"
            )

        take = min(remaining, 10000, width)
        indices = rng.choice(width, take, replace=False)

        chunks.append(
            {key: np.asarray(data[key])[indices]
             for key in BRANCHES}
        )
        remaining -= take

    if remaining:
        print(f"WARNING: {sample}: {remaining} fewer events")

    merged = {
        key: np.concatenate([chunk[key] for chunk in chunks])
        for key in BRANCHES
    }
    data_by_sample[sample] = merged
    print(f"Loaded {sample}: {len(merged['m_bb'])} events")

def plot_candidate(mass_key, flag_key, target, bins, filename, title):
    fig, ax = plt.subplots(figsize=(8, 5))

    for sample in SAMPLES:
        d = data_by_sample[sample]
        valid = (
            (d[flag_key] != 0)
            & np.isfinite(d[mass_key])
            & (d[mass_key] >= 0)
        )
        masses = d[mass_key][valid]

        if len(masses) == 0:
            continue

        ax.hist(
            masses, bins=bins,
            weights=np.full(len(masses), 1.0/len(masses)),
            histtype="step", linewidth=1.8,
            label=f"{sample} (n={len(masses)})"
        )

    ax.axvline(target, color="black", linestyle="--",
               label=f"Target mass: {target} GeV")
    ax.set_xlabel("Candidate invariant mass [GeV]")
    ax.set_ylabel("Fraction per bin (among candidates)")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / filename, dpi=180)
    plt.close(fig)

plot_candidate(
    "m_bb", "has_bb_candidate", 125.0,
    np.linspace(0, 300, 101),
    "m_bb.png", "Higgs candidate mass"
)

plot_candidate(
    "m_W_had", "has_jj_candidate", 80.4,
    np.linspace(0, 200, 101),
    "m_W_had.png", "W candidate mass"
)

print("\n=== Candidate availability and mass windows ===")
print("Fractions below are unweighted.")

for sample in SAMPLES:
    d = data_by_sample[sample]
    total = len(d["m_bb"])

    print(f"\n[{sample}] Total sampled events: {total}")

    for mass_key, flag_key, target, half_window in [
        ("m_bb", "has_bb_candidate", 125.0, 15.0),
        ("m_W_had", "has_jj_candidate", 80.4, 15.0)
    ]:
        valid = (
            (d[flag_key] != 0)
            & np.isfinite(d[mass_key])
            & (d[mass_key] >= 0)
        )
        nvalid = int(np.sum(valid))
        near = (
            valid
            & (np.abs(d[mass_key]-target) < half_window)
        )
        nnear = int(np.sum(near))

        availability = nvalid / total if total else 0
        conditional = nnear / nvalid if nvalid else 0
        overall = nnear / total if total else 0

        print(f"  {mass_key}:")
        print(f"    candidate availability: {availability:.4f}")
        print(f"    near mass |dm|<15 GeV / candidates: "
              f"{conditional:.4f}")
        print(f"    near mass |dm|<15 GeV / all: "
              f"{overall:.4f}")

print("\nSUCCESS: Reconstruction EDA completed")
