#!/usr/bin/env python3
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import argparse
import ast
import csv
import json
from pathlib import Path

import awkward as ak
import numpy as np
import tensorflow as tf
import uproot

SAMPLES = ["ttHH", "HH", "ttbar"]
SHARDS_PER_SAMPLE = 20
CHUNK = 10000

OUT = Path("results/dnn_weighted_efficiency")
PARTIAL = OUT / "partials"
PARTIAL.mkdir(parents=True, exist_ok=True)

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
CUTS = {
    "90": 0.267144,
    "80": 0.561638,
    "70": 0.765470,
    "50": 0.937753,
}

# Whole-file generator sums from previously verified audit.
AUDIT_SUMW = {
    "ttHH": 9987962.0,
    "HH": 35550.602502387017,
    "ttbar": 37820110815.09285,
}
AUDIT_COUNT = {
    "ttHH": 9987962,
    "HH": 1199991,
    "ttbar": 470123263,
}

def load_physics_functions():
    tree = ast.parse(
        Path("scripts/train_allpairs.py").read_text()
    )
    required = {"delta_r", "selected_jets", "pair_stats"}
    nodes = [
        node for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name in required
    ]
    if {node.name for node in nodes} != required:
        raise RuntimeError("Missing physics functions")

    scope = {"np": np, "ak": ak}
    exec(
        compile(
            ast.Module(body=nodes, type_ignores=[]),
            "train_allpairs.py", "exec"
        ),
        scope
    )
    return scope["selected_jets"], scope["pair_stats"]


def empty_stats():
    return {
        "n_total": 0,
        "sumw_total": 0.0,
        "sumw2_total": 0.0,
        "cuts": {
            key: {
                "n_pass": 0,
                "sumw_pass": 0.0,
                "sumw2_pass": 0.0,
            }
            for key in CUTS
        }
    }


def evaluate_task(task):
    sample_id = task // SHARDS_PER_SAMPLE
    shard_id = task % SHARDS_PER_SAMPLE
    sample = SAMPLES[sample_id]

    with open(f"config/filelists/{sample}.txt") as f:
        files = [
            line.strip() for line in f
            if line.strip()
        ]

    # Deterministic, non-overlapping file assignment.
    files = [
        filename
        for idx, filename in enumerate(files)
        if idx % SHARDS_PER_SAMPLE == shard_id
    ]
    if not files:
        raise RuntimeError("Empty shard")

    print(
        f"Sample={sample}, shard={shard_id}, "
        f"files={len(files)}",
        flush=True
    )

    selected_jets, pair_stats = load_physics_functions()

    scaler = np.load(
        "results/dnn_allpairs/C_all_pairs_scaler.npz"
    )
    if list(scaler["features"]) != FEATURES:
        raise RuntimeError("Feature order mismatch")

    mean = scaler["mean"]
    scale = scaler["scale"]

    model = tf.keras.models.load_model(
        "results/dnn_allpairs/C_all_pairs.keras",
        compile=False
    )

    stats = empty_stats()

    for file_idx, original in enumerate(files, 1):
        # Reuse previously generated object/feature caches.
        # The cache filenames match original filelist indices.
        with open(f"config/filelists/{sample}.txt") as f:
            allfiles = [
                line.strip() for line in f
                if line.strip()
            ]

        original_idx = allfiles.index(original)

        feature_path = (
            Path("cache/features") / sample
            / f"{original_idx:04d}.root"
        )
        object_path = (
            Path("cache/objects") / sample
            / f"{original_idx:04d}.root"
        )

        with uproot.open(feature_path) as fr, \
             uproot.open(object_path) as ob:

            ft = fr["Events"]
            ot = ob["Events"]

            if ft.num_entries != ot.num_entries:
                raise RuntimeError(
                    f"Event count mismatch: {feature_path}"
                )

            nentries = ft.num_entries

            for start in range(0, nentries, CHUNK):
                stop = min(start + CHUNK, nentries)

                f = ft.arrays(
                    BASIC + ["nNonBJets", "genWeight"],
                    entry_start=start,
                    entry_stop=stop,
                    library="np"
                )
                o = ot.arrays(
                    OBJECT,
                    entry_start=start,
                    entry_stop=stop,
                    library="ak"
                )

                # Guard against misaligned caches.
                for key in ("run", "luminosityBlock", "event"):
                    if key in ft and key in ot:
                        a = ft[key].array(
                            entry_start=start,
                            entry_stop=stop,
                            library="np"
                        )
                        b = ot[key].array(
                            entry_start=start,
                            entry_stop=stop,
                            library="np"
                        )
                        if not np.array_equal(a, b):
                            raise RuntimeError(
                                f"Event ID mismatch: "
                                f"{feature_path}, {start}"
                            )

                jets, bjets, nonbjets = selected_jets(o)

                counts = np.column_stack([
                    ak.to_numpy(ak.num(jets)),
                    ak.to_numpy(ak.num(bjets)),
                    ak.to_numpy(ak.num(nonbjets))
                ])
                expected = np.column_stack([
                    f["nSelectedJet"],
                    f["nBJets"],
                    f["nNonBJets"]
                ])
                if not np.array_equal(counts, expected):
                    bad = np.flatnonzero(
                        np.any(counts != expected, axis=1)
                    )

                    # Explicitly documented numerical boundary case:
                    # ttbar/0048.root, entry 694406.
                    known = (
                        str(feature_path)
                        == "cache/features/ttbar/0048.root"
                        and len(bad) == 1
                        and start + int(bad[0]) == 694406
                        and counts[bad[0]].tolist() == [2, 1, 1]
                        and expected[bad[0]].tolist() == [3, 1, 2]
                    )

                    if not known:
                        raise RuntimeError(
                            f"Unexpected Jet count mismatch: "
                            f"{feature_path}, {start}, "
                            f"entries={bad.tolist()}"
                        )

                    print(
                        "KNOWN BOUNDARY EXCEPTION: "
                        f"{feature_path}, entry=694406; "
                        "C++ DeltaR=0.400000005960, "
                        "Python DeltaR≈0.399999826",
                        flush=True
                    )

                x = np.column_stack([
                    *[f[key] for key in BASIC],
                    pair_stats(bjets),
                    pair_stats(nonbjets)
                ]).astype(np.float32)

                if x.shape[1] != 20:
                    raise RuntimeError(
                        f"Incorrect feature shape: {x.shape}"
                    )
                if not np.isfinite(x).all():
                    raise RuntimeError(
                        f"Non-finite features: {feature_path}"
                    )

                x = (
                    (x.astype(np.float64) - mean) / scale
                ).astype(np.float32)

                score = model(
                    x, training=False
                ).numpy()[:, 0]

                w = np.asarray(
                    f["genWeight"], dtype=np.float64
                )
                if not np.isfinite(w).all():
                    raise RuntimeError(
                        f"Non-finite genWeight: {feature_path}"
                    )

                stats["n_total"] += len(w)
                stats["sumw_total"] += float(np.sum(w))
                stats["sumw2_total"] += float(np.dot(w, w))

                for key, cut in CUTS.items():
                    passed = score > cut
                    pw = w[passed]
                    rec = stats["cuts"][key]
                    rec["n_pass"] += int(np.count_nonzero(passed))
                    rec["sumw_pass"] += float(np.sum(pw))
                    rec["sumw2_pass"] += float(np.dot(pw, pw))

        print(
            f"{sample} shard={shard_id}: "
            f"{file_idx}/{len(files)} files",
            flush=True
        )

    result = {
        "sample": sample,
        "shard_id": shard_id,
        "files_processed": len(files),
        "stats": stats
    }
    dest = PARTIAL / f"{sample}_{shard_id:02d}.json"
    tmp = dest.with_suffix(".tmp")
    tmp.write_text(json.dumps(result, indent=2))
    tmp.replace(dest)

    print(f"SUCCESS {dest}", flush=True)


def reduce_results():
    rows = []

    for sample in SAMPLES:
        combined = empty_stats()

        for shard in range(SHARDS_PER_SAMPLE):
            path = PARTIAL / f"{sample}_{shard:02d}.json"
            if not path.exists():
                raise RuntimeError(f"Missing result: {path}")

            result = json.loads(path.read_text())
            if (result["sample"] != sample
                    or result["shard_id"] != shard):
                raise RuntimeError(f"Wrong shard: {path}")

            part = result["stats"]
            for key in (
                "n_total", "sumw_total", "sumw2_total"
            ):
                combined[key] += part[key]

            for cut_id in CUTS:
                for key in (
                    "n_pass", "sumw_pass", "sumw2_pass"
                ):
                    combined["cuts"][cut_id][key] += (
                        part["cuts"][cut_id][key]
                    )

        n = combined["n_total"]
        denom = combined["sumw_total"]

        if n != AUDIT_COUNT[sample]:
            raise RuntimeError(
                f"{sample}: count mismatch "
                f"{n} vs {AUDIT_COUNT[sample]}"
            )

        rel = abs(
            denom - AUDIT_SUMW[sample]
        ) / max(1.0, abs(AUDIT_SUMW[sample]))
        if rel > 1e-6:
            raise RuntimeError(
                f"{sample}: sumw mismatch, rel={rel}"
            )

        print(
            f"{sample}: events={n}, "
            f"sumw={denom:.8f}, audit OK"
        )

        for cut_id, threshold in CUTS.items():
            rec = combined["cuts"][cut_id]
            weighted = rec["sumw_pass"] / denom
            unweighted = rec["n_pass"] / n

            rows.append({
                "sample": sample,
                "target_from_test": cut_id,
                "threshold": threshold,
                "n_total": n,
                "n_pass": rec["n_pass"],
                "sumw_total": denom,
                "sumw_pass": rec["sumw_pass"],
                "sumw2_total": combined["sumw2_total"],
                "sumw2_pass": rec["sumw2_pass"],
                "unweighted_eff": unweighted,
                "weighted_eff": weighted,
                "weighted_rejection": 1.0-weighted
            })

    output = OUT / "fullmc_efficiency.csv"
    with output.open("w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=list(rows[0])
        )
        writer.writeheader()
        writer.writerows(rows)

    print("\n=== FULL MC WEIGHTED EFFICIENCY ===")
    print(
        "sample | test target | cut | "
        "unweighted eff | weighted eff | weighted rejection"
    )

    for r in rows:
        print(
            f"{r['sample']:5s} | "
            f"{r['target_from_test']:>3s}% | "
            f"{r['threshold']:.6f} | "
            f"{r['unweighted_eff']:.6%} | "
            f"{r['weighted_eff']:.6%} | "
            f"{r['weighted_rejection']:.6%}"
        )

    print(f"\nSUCCESS: {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--task", type=int)
    group.add_argument("--reduce", action="store_true")
    args = parser.parse_args()

    if args.reduce:
        reduce_results()
    else:
        if not 0 <= args.task < (
            len(SAMPLES)*SHARDS_PER_SAMPLE
        ):
            raise ValueError("Task index must be 0..59")
        evaluate_task(args.task)
