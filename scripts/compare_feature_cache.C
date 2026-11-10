#include <ROOT/RDataFrame.hxx>
#include <TChain.h>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

int compare_feature_cache()
{
    TChain cached("Events");
    TChain direct("Events");

    cached.Add("cache/features/ttHH/test_features.root");
    direct.Add("cache/features/ttHH/test_features_direct.root");

    if (cached.GetEntries() != direct.GetEntries())
        throw std::runtime_error("Event counts differ");

    cached.AddFriend(&direct, "direct");

    ROOT::RDataFrame df(cached);

    const std::vector<std::string> fields = {
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

    // Compare with relative tolerance 1e-5.
    std::string mismatchExpr;

    for (const auto& f : fields) {
        const std::string a = "double(" + f + ")";
        const std::string b = "double(direct." + f + ")";

        const std::string expr =
            "(abs(" + a + "-" + b + ") > "
            "1e-5 * (1.0 + abs(" + b + ")))";

        if (!mismatchExpr.empty())
            mismatchExpr += " || ";

        mismatchExpr += expr;
    }

    auto mismatchCount = df.Filter(mismatchExpr).Count();

    const auto mismatches = *mismatchCount;

    std::cout << "Total events: "
              << cached.GetEntries() << "\n";
    std::cout << "Mismatched events: "
              << mismatches << "\n";

    if (mismatches != 0)
        throw std::runtime_error(
            "Feature comparison failed");

    std::cout << "SUCCESS: All compared features agree\n";
    return 0;
}
