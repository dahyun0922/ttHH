#pragma once

#include <ROOT/RVec.hxx>
#include <Math/GenVector/LorentzVector.h>
#include <Math/GenVector/PtEtaPhiM4D.h>
#include <algorithm>
#include <cmath>

using namespace ROOT;
using namespace ROOT::VecOps;

inline constexpr double kHiggsMass = 125.0;
inline constexpr double kWMass = 80.4;

inline double Reconstructor(double targetMass, const RVecF &pt, const RVecF &eta, const RVecF &phi, const RVecF &mass) {
   auto idx_cmb = Combinations(pt, 2);
   float bestM = -1;
   double bestDiff = 1e9;
   for (size_t i = 0; i < idx_cmb[0].size(); i++) {
      const auto i1 = idx_cmb[0][i];
      const auto i2 = idx_cmb[1][i];
      ROOT::Math::PtEtaPhiMVector p1(pt[i1], eta[i1], phi[i1], mass[i1]);
      ROOT::Math::PtEtaPhiMVector p2(pt[i2], eta[i2], phi[i2], mass[i2]);
      double m = (p1 + p2).M();
      if (std::abs(m - targetMass) < bestDiff) {
         bestDiff = std::abs(m - targetMass);
         bestM = m;
      }
   }
   return bestM;
}


// Returns:
// [0] mass of the two leading-pT b-jets
// [1] pT of the closest-to-125-GeV bb candidate
// [2] DeltaR of that candidate
// [3] reconstructed candidate mass (cross-check)
inline RVecF BBExtra(const RVecF &pt,
              const RVecF &eta,
              const RVecF &phi,
              const RVecF &mass) {
   RVecF result(4, -1.f);
   if (pt.size() < 2) return result;

   auto idx = Reverse(Argsort(pt));

   auto p4 = [&](size_t i) {
      return ROOT::Math::PtEtaPhiMVector(
          pt[i], eta[i], phi[i], mass[i]);
   };

   // Leading-pT pair
   result[0] = (p4(idx[0]) + p4(idx[1])).M();

   // Reproduce the original closest-mass pairing
   double bestDiff = 1e9;
   size_t bestI = 0, bestJ = 1;

   for (size_t i = 0; i < pt.size(); ++i) {
      for (size_t j = i + 1; j < pt.size(); ++j) {
         double m = (p4(i) + p4(j)).M();
         double diff = std::abs(m - kHiggsMass);

         if (diff < bestDiff) {
            bestDiff = diff;
            bestI = i;
            bestJ = j;
         }
      }
   }

   auto candidate = p4(bestI) + p4(bestJ);

   result[1] = candidate.Pt();
   result[2] = ROOT::VecOps::DeltaR(
       eta[bestI], eta[bestJ],
       phi[bestI], phi[bestJ]);
   result[3] = candidate.M();

   return result;
}
