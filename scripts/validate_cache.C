#include <ROOT/RDataFrame.hxx>
#include <TChain.h>

#include <cmath>
#include <cstdio>
#include <iostream>
#include <stdexcept>
#include <string>

int validate_cache(const char* sample,
                   int expectedFiles,
                   long long expectedEvents,
                   double expectedSumW)
{
    for (const auto& level : {"objects", "features"}) {
        TChain chain("Events");

        for (int i = 0; i < expectedFiles; ++i) {
            char path[512];
            std::snprintf(
                path, sizeof(path),
                "cache/%s/%s/%04d.root",
                level, sample, i
            );

            if (chain.Add(path) == 0)
                throw std::runtime_error(
                    std::string("Missing file: ") + path);
        }

        ROOT::RDataFrame df(chain);

        auto weighted = df.Define(
            "validationWeight",
            "static_cast<double>(genWeight)"
        );

        auto count = weighted.Count();
        auto sumW = weighted.Sum<double>("validationWeight");

        const auto n = *count;
        const auto w = *sumW;

        std::cout.precision(17);
        std::cout << sample << " / " << level << "\n";
        std::cout << "  Events: " << n << "\n";
        std::cout << "  SumW: " << w << "\n";

        const double tolerance =
            1e-5 * std::max(1.0, std::abs(expectedSumW));

        if (n != static_cast<ULong64_t>(expectedEvents))
            throw std::runtime_error("Event count mismatch");

        if (!std::isfinite(w) ||
            std::abs(w - expectedSumW) > tolerance)
            throw std::runtime_error("Weight sum mismatch");

        std::cout << "  VALIDATION PASSED\n";
    }

    return 0;
}
