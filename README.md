# ttHH Classification with DNN

CMS Run III 2024 NanoAOD MC를 사용해
ttHH(signal), HH, ttbar(background)를 분류하는
TensorFlow 기반 DNN 분석 프로젝트입니다.

자세한 연구 과정과 물리적 해석은
[연구 요약](ttHH_DNN_analysis_summary.md)을 참고하세요.

## 1. Dataset

| Class | Process | Files | Events |
|---|---|---:|---:|
| 0 | ttHH | 220 | 9,987,962 |
| 1 | HH | 61 | 1,199,991 |
| 2 | ttbar | 780 | 470,123,263 |

- Collision energy: 13.6 TeV
- Campaign: RunIII2024Summer24NanoAODv15
- Input: CMS NanoAODSIM
- ROOT object/feature caches are not distributed through Git.

The training uses balanced MC samples without event weights.
Generator weights are used separately for full-MC efficiency studies.

## 2. Environment

Main dependencies:
- Python 3.9
- TensorFlow
- NumPy, Awkward Array, uproot
- scikit-learn, Matplotlib
- ROOT (for cache generation and validation)
- Slurm (for batch jobs)

Example setup:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install tensorflow numpy awkward uproot scikit-learn matplotlib
```

The original NanoAOD files must be available separately.
Sample filelists and cache paths need to match the local environment.

## 3. Analysis workflow

### Step 1: Baseline DNN

```bash
sbatch train_dnn.sh
```

- Script: `scripts/train_baseline.py`
- Input: 8 event-level features
- Architecture: 64 -> 64 -> 32 -> softmax(3)
- Output: `results/dnn_baseline/`
- Baseline ttHH ROC-AUC: approximately 0.944

### Step 2: All-pairs DNN

```bash
sbatch train_allpairs.sh
```

- Script: `scripts/train_allpairs.py`
- Input: 8 basic + 12 jet-pair features
- b-jet and non-b-jet pairs are considered separately.
- Pair mass and deltaR: min, mean, max
- No Higgs/W closest-mass candidate selection
- Output: `results/dnn_allpairs/`
- ttHH ROC-AUC: approximately 0.952

### Step 3: Feature validation

| Script | Purpose |
|---|---|
| `feature_importance.py` | Permutation importance |
| `ablation_st.py` | ST ablation |
| `compare_reco_features.py` | Reconstructed-candidate feature comparison |
| `compare_pair_groups.py` | Mass vs deltaR feature ablation |
| `evaluate_multiplicity.py` | Performance at fixed jet multiplicity |

Run through the matching Slurm shell scripts when available.

### Step 4: Multi-seed validation

```bash
sbatch validate_multiseed.sh
```

- Script: `scripts/validate_multiseed.py`
- Seeds: 42, 123, 2026, 777, 31415
- Fixed test sample
- All-pairs AUC improvement: +0.00681
- Run-to-run standard deviation: 0.00017
- Improvement observed in all 5 runs

### Step 5: DNN score efficiency

```bash
sbatch evaluate_dnn_efficiency.sh
```

- Script: `scripts/evaluate_dnn_efficiency.py`
- Uses a previously trained All-pairs model
- Calculates unweighted signal/background efficiency
- Output: `results/dnn_efficiency/`

### Step 6: Full-MC weighted efficiency

```bash
sbatch evaluate_weighted_fullmc.sh
```

- Script: `scripts/evaluate_weighted_fullmc.py`
- 60 Slurm array tasks
- Processes all 481,311,216 cached MC events
- Uses signed generator weights
- Partial results: `results/dnn_weighted_efficiency/partials/`

After all array tasks finish successfully:

```bash
python scripts/evaluate_weighted_fullmc.py --reduce
```

Output:
`results/dnn_weighted_efficiency/fullmc_efficiency.csv`

## 4. Main results

For a common ttHH DNN score threshold of 0.561638:

| Process | Weighted pass efficiency |
|---|---:|
| ttHH | 80.4277% |
| HH | 3.4985% |
| ttbar | 6.5460% |

The corresponding background rejection rates are
96.5015% (HH) and 93.4540% (ttbar).

These results describe the three investigated MC samples,
not the complete background composition of CMS collisions.

## 5. Reproducibility notes

- The trained Keras models and large ROOT caches are not
  included in the repository.
- Scripts requiring trained models must be run after training.
- Some paths are specific to the original HPC environment.
- The full-MC weighted evaluation includes training events;
  it is not an independent generalization test.
- One known DeltaR=0.4 numerical boundary mismatch between
  ROOT C++ and Python is explicitly handled in the
  full-MC evaluation script. This is a temporary exception.
- The cross-section-normalized physics yields are preliminary:
  decay branching fractions and generator normalization
  require further verification.

## 6. Next steps

1. Harmonize C++ and Python jet selections.
2. Validate sample cross sections and decay branching fractions.
3. Define physics-motivated signal/control regions.
4. Include additional backgrounds and systematic uncertainties.
5. Test on statistically independent MC and collision data.
