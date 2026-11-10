#include <ROOT/RDataFrame.hxx>
#include <TFile.h>
#include <TTree.h>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

int make_object_cache(const char* inputFile,
                      const char* outputFile)
{
    ROOT::RDataFrame df("Events", inputFile);

    std::vector<std::string> columns = {
        "Jet_pt",
        "Jet_eta",
        "Jet_phi",
        "Jet_mass",
        "Jet_btagUParTAK4B",

        "Muon_pt",
        "Muon_eta",
        "Muon_phi",
        "Muon_mediumId",
        "Muon_pfRelIso04_all",

        "Electron_pt",
        "Electron_eta",
        "Electron_phi",
        "Electron_cutBased",

        "PuppiMET_pt",
        "genWeight"
    };

    // Preserve event identifiers if available.
    for (const auto& name :
         {"run", "luminosityBlock", "event"}) {
        if (df.HasColumn(name))
            columns.push_back(name);
    }

    for (const auto& name : columns) {
        if (!df.HasColumn(name))
            throw std::runtime_error(
                "Missing branch: " + name);
    }

    std::cout << "=== Object cache ===\n";
    std::cout << "Input: " << inputFile << "\n";
    std::cout << "Output: " << outputFile << "\n";

    // Snapshot keeps the original event-level arrays.
    // No filters or object cuts are applied.
    df.Snapshot("Events", outputFile, columns);

    TFile source(inputFile, "READ");
    TFile cached(outputFile, "READ");

    if (source.IsZombie() || cached.IsZombie())
        throw std::runtime_error("File verification failed");

    auto* originalTree = source.Get<TTree>("Events");
    auto* cachedTree = cached.Get<TTree>("Events");

    if (!originalTree || !cachedTree)
        throw std::runtime_error("Events tree missing");

    auto originalN = originalTree->GetEntries();
    auto cachedN = cachedTree->GetEntries();

    std::cout << "Original events: " << originalN << "\n";
    std::cout << "Cached events: " << cachedN << "\n";

    if (originalN != cachedN)
        throw std::runtime_error("Event count mismatch");

    for (const auto& name : columns) {
        if (!cachedTree->GetBranch(name.c_str()))
            throw std::runtime_error(
                "Cached branch missing: " + name);
    }

    std::cout << "Branches preserved: "
              << columns.size() << "\n";
    std::cout << "SUCCESS: Object cache verified\n";

    return 0;
}
