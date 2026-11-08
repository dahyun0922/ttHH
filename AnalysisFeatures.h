#pragma once

#include "Selection.h"
#include "Reconstruction.h"

#include <ROOT/RDataFrame.hxx>
#include <ROOT/RVec.hxx>
#include <algorithm>

using namespace ROOT;
using namespace ROOT::VecOps;

inline ROOT::RDF::RNode AddBJetVars(ROOT::RDF::RNode df) {
   return df
             .Define("muMask",
                 Selection::MuonMask<int>,
                 {"Muon_pt", "Muon_eta",
                  "Muon_mediumId",
                  "Muon_pfRelIso04_all"})

             .Define("elMask",
                 Selection::ElectronMask<int>,
                 {"Electron_pt", "Electron_eta",
                  "Electron_cutBased"})

             .Define("selectedMuon_pt",
                 "Muon_pt[muMask]")
             .Define("selectedMuon_eta",
                 "Muon_eta[muMask]")
             .Define("selectedMuon_phi",
                 "Muon_phi[muMask]")

             .Define("selectedElectron_pt",
                 "Electron_pt[elMask]")
             .Define("selectedElectron_eta",
                 "Electron_eta[elMask]")
             .Define("selectedElectron_phi",
                 "Electron_phi[elMask]")

             .Define("jetMask",
                 Selection::JetMask<int>,
                 {"Jet_pt", "Jet_eta", "Jet_phi",
                  "Jet_jetId",
                  "selectedMuon_eta",
                  "selectedMuon_phi",
                  "selectedElectron_eta",
                  "selectedElectron_phi"})

             .Define("selectedJet_pt",
                 "Jet_pt[jetMask]")
             .Define("selectedJet_eta",
                 "Jet_eta[jetMask]")
             .Define("selectedJet_phi",
                 "Jet_phi[jetMask]")
             .Define("selectedJet_mass",
                 "Jet_mass[jetMask]")
             .Define("selectedJet_btagUParTAK4B",
                 "Jet_btagUParTAK4B[jetMask]")

             .Define("nSelectedJet",
                 "static_cast<int>(selectedJet_pt.size())")

             .Define("HT", "Sum(selectedJet_pt)")
            .Define("ST", "HT + Sum(selectedMuon_pt) + Sum(selectedElectron_pt) + PuppiMET_pt")
            .Define("leading_jet_pt",
                "selectedJet_pt.size() > 0 ? Max(selectedJet_pt) : -1.f")
            .Define("subleading_jet_pt",
                "selectedJet_pt.size() > 1 ? Reverse(Sort(selectedJet_pt))[1] : -1.f")
            .Define("leading_lepton_pt",
                [](const RVecF &mu, const RVecF &el) {
                   float muMax = mu.empty() ? -1.f : Max(mu);
                   float elMax = el.empty() ? -1.f : Max(el);
                   return std::max(muMax, elMax);
                },
                {"selectedMuon_pt", "selectedElectron_pt"})
            .Define("isBJet",     "selectedJet_btagUParTAK4B > 0.1272")
            .Define("nBJets",     "Sum(isBJet)")
             .Define("has_bb_candidate",
                 "nBJets >= 2")
             .Define("has_jj_candidate",
                 "nSelectedJet - nBJets >= 2")
            .Define("bJet_pt",    "selectedJet_pt[isBJet]")
            .Define("bJet_eta",   "selectedJet_eta[isBJet]")
            .Define("bJet_phi",   "selectedJet_phi[isBJet]")
            .Define("bJet_mass",  "selectedJet_mass[isBJet]")
            .Define("deltaR_bb_leading",
                [](const RVecF &pt,
                   const RVecF &eta,
                   const RVecF &phi) {
                   if (pt.size() < 2) return -1.0;

                   auto idx = Reverse(Argsort(pt));
                   return static_cast<double>(
                       ROOT::VecOps::DeltaR(
                           eta[idx[0]], eta[idx[1]],
                           phi[idx[0]], phi[idx[1]]
                       )
                   );
                },
                {"bJet_pt", "bJet_eta", "bJet_phi"})

            .Define("bb_extra", BBExtra,
                {"bJet_pt", "bJet_eta",
                 "bJet_phi", "bJet_mass"})
            .Define("m_bb_leading", "bb_extra[0]")
            .Define("pt_bb", "bb_extra[1]")
            .Define("deltaR_bb_Hcand", "bb_extra[2]")
            .Define("m_bb_closest_check", "bb_extra[3]")
            .Define("isNotBJet",  "!isBJet")
            .Define("lJet_pt",    "selectedJet_pt[isNotBJet]")
            .Define("lJet_eta",   "selectedJet_eta[isNotBJet]")
            .Define("lJet_phi",   "selectedJet_phi[isNotBJet]")
            .Define("lJet_mass",  "selectedJet_mass[isNotBJet]");
}

inline ROOT::RDF::RNode Mbb(ROOT::RDF::RNode df) {
   return df.Filter("bJet_pt.size() >= 2", "At least two b-tagged jets")
            .Define("m_bb",
                [](const RVecF &pt, const RVecF &eta, const RVecF &phi, const RVecF &mass) {
                   return Reconstructor(kHiggsMass, pt, eta, phi, mass);
                },
                {"bJet_pt", "bJet_eta", "bJet_phi", "bJet_mass"});
}

inline ROOT::RDF::RNode WHad(ROOT::RDF::RNode df) {
   return df.Filter("lJet_pt.size() >= 2", "At least two non-b-tagged jets")
            .Define("m_W_had",
                [](const RVecF &pt, const RVecF &eta, const RVecF &phi, const RVecF &mass) {
                   return Reconstructor(kWMass, pt, eta, phi, mass);
                },
                {"lJet_pt", "lJet_eta", "lJet_phi", "lJet_mass"});
}
