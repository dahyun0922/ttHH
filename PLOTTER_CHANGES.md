# Plotter.C Modification Summary

## Overview

This document summarizes the modifications made to the original
`Plotter.C` from the GitHub repository:

https://github.com/dahyun0922/ttHH

The goal is to extend the original EDA code toward a physics-based
DNN classification analysis of ttHH, HH, and ttbar events.

## 1. Additional Kinematic Features

The original Plotter produced 7 histograms.
The modified version produces 16 histograms.

| Added Feature | Description |
|---|---|
| HT | Scalar sum of selected jet transverse momenta |
| ST | HT + selected lepton pT sum + MET |
| leading_jet_pt | Highest-pT selected jet |
| subleading_jet_pt | Second-highest-pT selected jet |
| leading_lepton_pt | Highest-pT selected muon or electron |
| deltaR_bb_leading | DeltaR between two leading-pT b-jets |
| m_bb_leading | Invariant mass of two leading-pT b-jets |
| pt_bb | Transverse momentum of the closest-to-125-GeV bb pair |
| deltaR_bb_Hcand | DeltaR of the closest-to-125-GeV bb pair |

The additional `m_bb_closest_check` column is used for
cross-checking the original reconstruction algorithm.

## 2. Object Selection

Added `Selection.h` to implement common object selection.

### Jets
- pT > 30 GeV
- |eta| < 2.4
- Tight Jet ID
- Remove jets overlapping with selected leptons (DeltaR < 0.4)

### Muons
- pT > 15 GeV
- |eta| < 2.4
- Medium ID
- Relative isolation < 0.25

### Electrons
- pT > 15 GeV
- |eta| < 2.5
- Medium cut-based ID (cutBased >= 3)

Selected object collections are used for downstream
feature calculations. Original NanoAOD branches are preserved.

## 3. Changes to AddBJetVars()

Originally, `AddBJetVars()` categorized raw jets into
b-tagged and non-b-tagged collections.

The modified function:
- Applies muon and electron selection.
- Performs jet selection and lepton-jet cleaning.
- Creates selected jet and lepton collections.
- Calculates HT, ST, and leading-object features.
- Performs b-tagging on selected jets.
- Calculates additional bb reconstruction observables.

The original b-tag threshold (0.1272) is retained
pending validation against the relevant CMS calibration.

## 4. bb Reconstruction

Added `BBExtra()` for additional bb observables.

The original `Reconstructor()` is retained.

Two pairing strategies are now available:
- Leading-pT pair: two highest-pT b-jets.
- Higgs candidate pair: bb pair closest to 125 GeV.

The modified code also defines:
- `has_bb_candidate`: at least two selected b-jets.
- `has_jj_candidate`: at least two selected non-b-jets.

The original `Mbb()` and `WHad()` filtering logic is retained,
but reconstruction now uses selected jet collections.

## 5. Existing Histogram Changes

The original histograms remain, but several now use
selected objects instead of raw NanoAOD collections.

| Histogram | Modification |
|---|---|
| Muon pT | Uses selected muons |
| Jet pT | Uses selected jets |
| nJet | Uses nSelectedJet |
| nBJets | Counts selected b-jets |
| MET | Unchanged |
| m_bb | Reconstructed from selected b-jets |
| m_jj | Reconstructed from selected non-b-jets |

The nJet histogram range was also standardized to
20 bins (-0.5 to 19.5) for all three samples.

## 6. Dataset Configuration

Removed hard-coded dataset paths from `Plotter()`.

Added `ReadDatasetPaths()` and a configuration argument:

    Plotter(const char *configFile = "config/datasets.txt")

Dataset paths are now supplied through a text file,
allowing the same analysis code to run on different servers.

## 7. Testing and Validation

Added synthetic ROOT datasets and the following tests:

| Test | Purpose |
|---|---|
| test_leading.C | Leading jet and lepton features |
| test_bb.C | bb reconstruction variables |
| test_selection.C | Object selection functions |
| test_selection_rdf.C | Selection in RDataFrame |
| test_cutflow.C | Jet multiplicity and candidate counts |

Validation completed:
- C++ syntax checks passed.
- ROOT RDataFrame execution passed.
- Synthetic dataset tests passed.
- All 16 histogram pages were generated successfully.
- Selection cutflow tests passed.

## 8. Remaining Work

- Validate NanoAOD branch types and CMS working points.
- Validate object selection with real MC datasets.
- Implement weighted event cutflow and efficiency tables.
- Handle missing candidates consistently for DNN training.
- Evaluate histogram normalization and overflow.
- Prepare event-by-event DNN input datasets.
- Add appropriate trigger and event-level selections.

## Notes

All current validation was performed using synthetic ROOT data.

The synthetic distributions are not representative of real
ttHH, HH, or ttbar physics distributions.

The modified code should not yet be considered a fully
validated CMS physics analysis.

## Code Refactoring

The analysis code was reorganized into reusable headers
without intentionally changing the physics calculations.

| File | Responsibility |
|---|---|
| Plotter.C | Histogram booking, normalization, and PDF output |
| Selection.h | Object selection and lepton-jet cleaning |
| Reconstruction.h | Reconstructor, BBExtra, and target masses |
| AnalysisFeatures.h | RDataFrame feature definitions and candidate filters |
| DatasetConfig.h | Dataset path configuration reader |

This separation keeps the plotting macro focused on EDA
and allows reconstruction and feature definitions to be
reused by future DNN dataset production code.
