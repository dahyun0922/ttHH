#ifndef TTHH_SELECTION_H
#define TTHH_SELECTION_H

#include <ROOT/RVec.hxx>
#include <cmath>
#include <stdexcept>

namespace Selection {

using ROOT::RVecF;
using ROOT::RVecB;
using namespace ROOT::VecOps;

// Muon: pT, eta, Medium ID, isolation
template <typename ID>
RVecB MuonMask(
    const RVecF &pt,
    const RVecF &eta,
    const ROOT::VecOps::RVec<ID> &mediumId,
    const RVecF &iso)
{
    if (pt.size() != eta.size() ||
        pt.size() != mediumId.size() ||
        pt.size() != iso.size())
        throw std::runtime_error("Muon size mismatch");

    RVecB mask(pt.size(), false);

    for (size_t i = 0; i < pt.size(); ++i) {
        mask[i] =
            pt[i] > 15.f &&
            std::abs(eta[i]) < 2.4f &&
            mediumId[i] != 0 &&
            iso[i] < 0.25f;
    }
    return mask;
}

// Electron: pT, eta, Medium cut-based ID
template <typename ID>
RVecB ElectronMask(
    const RVecF &pt,
    const RVecF &eta,
    const ROOT::VecOps::RVec<ID> &cutBased)
{
    if (pt.size() != eta.size() ||
        pt.size() != cutBased.size())
        throw std::runtime_error("Electron size mismatch");

    RVecB mask(pt.size(), false);

    for (size_t i = 0; i < pt.size(); ++i) {
        mask[i] =
            pt[i] > 15.f &&
            std::abs(eta[i]) < 2.5f &&
            cutBased[i] >= 3;
    }
    return mask;
}

// Jet: kinematics, Tight ID, lepton cleaning
template <typename ID>
RVecB JetMask(
    const RVecF &pt,
    const RVecF &eta,
    const RVecF &phi,
    const ROOT::VecOps::RVec<ID> &jetId,
    const RVecF &muEta,
    const RVecF &muPhi,
    const RVecF &elEta,
    const RVecF &elPhi)
{
    if (pt.size() != eta.size() ||
        pt.size() != phi.size() ||
        pt.size() != jetId.size() ||
        muEta.size() != muPhi.size() ||
        elEta.size() != elPhi.size())
        throw std::runtime_error("Object size mismatch");

    RVecB mask(pt.size(), false);

    for (size_t i = 0; i < pt.size(); ++i) {
        if (pt[i] <= 30.f ||
            std::abs(eta[i]) >= 2.4f ||
            (static_cast<int>(jetId[i]) & 2) == 0)
            continue;

        bool overlaps = false;

        for (size_t j = 0; j < muEta.size(); ++j) {
            if (DeltaR(eta[i], muEta[j],
                       phi[i], muPhi[j]) < 0.4f) {
                overlaps = true;
                break;
            }
        }

        if (overlaps) continue;

        for (size_t j = 0; j < elEta.size(); ++j) {
            if (DeltaR(eta[i], elEta[j],
                       phi[i], elPhi[j]) < 0.4f) {
                overlaps = true;
                break;
            }
        }

        mask[i] = !overlaps;
    }

    return mask;
}


// Preliminary jet selection for datasets without Jet_jetId.
// WARNING: Tight Jet ID is NOT applied.
inline RVecB JetMaskNoId(
    const RVecF &pt,
    const RVecF &eta,
    const RVecF &phi,
    const RVecF &muEta,
    const RVecF &muPhi,
    const RVecF &elEta,
    const RVecF &elPhi)
{
    if (pt.size() != eta.size() ||
        pt.size() != phi.size() ||
        muEta.size() != muPhi.size() ||
        elEta.size() != elPhi.size())
        throw std::runtime_error("Object size mismatch");

    RVecB mask(pt.size(), false);

    for (size_t i = 0; i < pt.size(); ++i) {
        if (pt[i] <= 30.f || std::abs(eta[i]) >= 2.4f)
            continue;

        bool overlap = false;

        for (size_t j = 0; j < muEta.size(); ++j) {
            if (DeltaR(eta[i], muEta[j],
                       phi[i], muPhi[j]) < 0.4f) {
                overlap = true;
                break;
            }
        }

        if (overlap) continue;

        for (size_t j = 0; j < elEta.size(); ++j) {
            if (DeltaR(eta[i], elEta[j],
                       phi[i], elPhi[j]) < 0.4f) {
                overlap = true;
                break;
            }
        }

        mask[i] = !overlap;
    }

    return mask;
}

} // namespace Selection

#endif
