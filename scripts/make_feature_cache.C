#include "../AnalysisFeatures.h"

#include <ROOT/RDataFrame.hxx>
#include <TFile.h>
#include <TTree.h>

#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

int make_feature_cache(const char* inputFile,
                       const char* outputFile,
                       int sampleId = 0)
{
    ROOT::RDataFrame df("Events", inputFile);

    auto features = AddBJetVars(df)
        .Define("nNonBJets",
                "nSelectedJet - nBJets")
        .Define("m_bb",
            [](const RVecF &pt,
               const RVecF &eta,
               const RVecF &phi,
               const RVecF &mass) {
                if (pt.size() < 2)
                    return -1.0;
                return Reconstructor(
                    kHiggsMass, pt, eta, phi, mass);
            },
            {"bJet_pt", "bJet_eta",
             "bJet_phi", "bJet_mass"})
        .Define("m_W_had",
            [](const RVecF &pt,
               const RVecF &eta,
               const RVecF &phi,
               const RVecF &mass) {
                if (pt.size() < 2)
                    return -1.0;
                return Reconstructor(
                    kWMass, pt, eta, phi, mass);
            },
            {"lJet_pt", "lJet_eta",
             "lJet_phi", "lJet_mass"})
        .Define("sample_id",
                [sampleId]() { return sampleId; });

    std::vector<std::string> columns = {
        "sample_id",
        "genWeight",

        "nSelectedJet",
        "nBJets",
        "nNonBJets",

        "HT",
        "ST",
        "PuppiMET_pt",

        "leading_jet_pt",
        "subleading_jet_pt",
        "leading_lepton_pt",

        "has_bb_candidate",
        "has_jj_candidate",

        "m_bb",
        "m_bb_leading",
        "pt_bb",
        "deltaR_bb_leading",
        "deltaR_bb_Hcand",

        "m_W_had"
    };

    for (const auto& name :
         {"run", "luminosityBlock", "event"}) {
        if (features.HasColumn(name))
            columns.push_back(name);
    }

    std::cout << "Input: " << inputFile << "\n";
    std::cout << "Output: " << outputFile << "\n";

    features.Snapshot("Events", outputFile, columns);

    TFile input(inputFile, "READ");
    TFile output(outputFile, "READ");

    if (input.IsZombie() || output.IsZombie())
        throw std::runtime_error("Cannot open ROOT file");

    auto* inTree = input.Get<TTree>("Events");
    auto* outTree = output.Get<TTree>("Events");

    if (!inTree || !outTree)
        throw std::runtime_error("Events tree missing");

    auto nInput = inTree->GetEntries();
    auto nOutput = outTree->GetEntries();

    std::cout << "Input events: " << nInput << "\n";
    std::cout << "Output events: " << nOutput << "\n";

    if (nInput != nOutput)
        throw std::runtime_error("Event count mismatch");

    for (const auto& name : columns) {
        if (!outTree->GetBranch(name.c_str()))
            throw std::runtime_error(
                "Missing feature: " + name);
    }

    std::cout << "Saved branches: "
              << columns.size() << "\n";
    std::cout << "SUCCESS: Feature cache verified\n";

    return 0;
}
