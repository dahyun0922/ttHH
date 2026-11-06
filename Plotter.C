#include <ROOT/RDataFrame.hxx>
#include <ROOT/RVec.hxx>
#include <ROOT/RDFHelpers.hxx>
#include <Math/Vector4Dfwd.h>
#include <Math/GenVector/LorentzVector.h>
#include <Math/GenVector/PtEtaPhiM4D.h>
#include <TCanvas.h>
#include <TLegend.h>
#include <TStyle.h>
#include <algorithm>

using namespace ROOT;
using namespace ROOT::VecOps;

const double kHiggsMass = 125.0;
const double kWMass = 80.4;

double Reconstructor(double targetMass, const RVecF &pt, const RVecF &eta, const RVecF &phi, const RVecF &mass) {
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

ROOT::RDF::RNode AddBJetVars(ROOT::RDF::RNode df) {
   return df.Define("isBJet",     "Jet_btagUParTAK4B > 0.1272")
            .Define("nBJets",     "Sum(isBJet)")
            .Define("bJet_pt",    "Jet_pt[isBJet]")
            .Define("bJet_eta",   "Jet_eta[isBJet]")
            .Define("bJet_phi",   "Jet_phi[isBJet]")
            .Define("bJet_mass",  "Jet_mass[isBJet]")
            .Define("isNotBJet",  "!isBJet")
            .Define("lJet_pt",    "Jet_pt[isNotBJet]")
            .Define("lJet_eta",   "Jet_eta[isNotBJet]")
            .Define("lJet_phi",   "Jet_phi[isNotBJet]")
            .Define("lJet_mass",  "Jet_mass[isNotBJet]");
}

ROOT::RDF::RNode Mbb(ROOT::RDF::RNode df) {
   return df.Filter("bJet_pt.size() >= 2", "At least two b-tagged jets")
            .Define("m_bb",
                [](const RVecF &pt, const RVecF &eta, const RVecF &phi, const RVecF &mass) {
                   return Reconstructor(kHiggsMass, pt, eta, phi, mass);
                },
                {"bJet_pt", "bJet_eta", "bJet_phi", "bJet_mass"});
}

ROOT::RDF::RNode WHad(ROOT::RDF::RNode df) {
   return df.Filter("lJet_pt.size() >= 2", "At least two non-b-tagged jets")
            .Define("m_W_had",
                [](const RVecF &pt, const RVecF &eta, const RVecF &phi, const RVecF &mass) {
                   return Reconstructor(kWMass, pt, eta, phi, mass);
                },
                {"lJet_pt", "lJet_eta", "lJet_phi", "lJet_mass"});
}

int Plotter() {
   ROOT::EnableImplicitMT();
   gStyle->SetOptStat(0);

   std::string dir_ttHH = "/data2/common/NanoAOD/mc/v15/RunIII2024Summer24NanoAODv15/TTHH-TTto2L2Nu-HHto2B2W_TuneCP5_13p6TeV_madgraph-pythia8/NANOAODSIM/150X_mcRun3_2024_realistic_v2-v2/*/*.root";

   std::string dir_HH = "/data2/common/NanoAOD/mc/v15/RunIII2024Summer24NanoAODv15/GluGlutoHHto2B2WtoLNu2Q_Par-c2-0p00-kl-1p00-kt-1p00_TuneCP5_13p6TeV_powheg-pythia8/NANOAODSIM/150X_mcRun3_2024_realistic_v2-v2/*/*.root";

   std::string dir_ttbar = "/data2/common/NanoAOD/mc/v15/RunIII2024Summer24NanoAODv15/TTto2L2Nu_TuneCP5_13p6TeV_powheg-pythia8/NANOAODSIM/150X_mcRun3_2024_realistic_v2-v3/*/*.root";

   auto df_ttHH  = AddBJetVars(ROOT::RDataFrame("Events", dir_ttHH));
   auto df_HH    = AddBJetVars(ROOT::RDataFrame("Events", dir_HH));
   auto df_ttbar = AddBJetVars(ROOT::RDataFrame("Events", dir_ttbar));

   auto hMuonPt_ttHH  = df_ttHH.Histo1D({"hMuonPt_ttHH", "Muon p_{T};Muon p_{T} [GeV];Normalized Events", 50, 0., 200.}, "Muon_pt", "genWeight");
   auto hJetPt_ttHH   = df_ttHH.Histo1D({"hJetPt_ttHH", "Jet p_{T};Jet p_{T} [GeV];Normalized Events", 50, 0., 300.}, "Jet_pt", "genWeight");
   auto hNJets_ttHH   = df_ttHH.Histo1D({"hNJets_ttHH", "Number of Jets;No. Jets;Normalized Events", 20, -0.5, 19.5}, "nJet", "genWeight");
   auto hNBJets_ttHH  = df_ttHH.Histo1D({"hNBJets_ttHH", "Number of b-Jets;No. b-Jets;Normalized Events", 6, -0.5, 5.5}, "nBJets", "genWeight");
   auto hMET_ttHH     = df_ttHH.Histo1D({"hMET_ttHH", "Missing Transverse Energy;MET [GeV];Normalized Events", 50, 0., 300.}, "PuppiMET_pt", "genWeight");

   auto hMuonPt_HH  = df_HH.Histo1D({"hMuonPt_HH", "Muon p_{T};Muon p_{T} [GeV];Normalized Events", 50, 0., 200.}, "Muon_pt", "genWeight");
   auto hJetPt_HH   = df_HH.Histo1D({"hJetPt_HH", "Jet p_{T};Jet p_{T} [GeV];Normalized Events", 50, 0., 300.}, "Jet_pt", "genWeight");
   auto hNJets_HH   = df_HH.Histo1D({"hNJets_HH", "Number of Jets;No. Jets;Normalized Events", 10, -0.5, 9.5}, "nJet", "genWeight");
   auto hNBJets_HH  = df_HH.Histo1D({"hNBJets_HH", "Number of b-Jets;No. b-Jets;Normalized Events", 6, -0.5, 5.5}, "nBJets", "genWeight");
   auto hMET_HH     = df_HH.Histo1D({"hMET_HH", "Missing Transverse Energy;MET [GeV];Normalized Events", 50, 0., 300.}, "PuppiMET_pt", "genWeight");

   auto hMuonPt_ttbar  = df_ttbar.Histo1D({"hMuonPt_ttbar", "Muon p_{T};Muon p_{T} [GeV];Normalized Events", 50, 0., 200.}, "Muon_pt", "genWeight");
   auto hJetPt_ttbar   = df_ttbar.Histo1D({"hJetPt_ttbar", "Jet p_{T};Jet p_{T} [GeV];Normalized Events", 50, 0., 300.}, "Jet_pt", "genWeight");
   auto hNJets_ttbar   = df_ttbar.Histo1D({"hNJets_ttbar", "Number of Jets;No. Jets;Normalized Events", 10, -0.5, 9.5}, "nJet", "genWeight");
   auto hNBJets_ttbar  = df_ttbar.Histo1D({"hNBJets_ttbar", "Number of b-Jets;No. b-Jets;Normalized Events", 6, -0.5, 5.5}, "nBJets", "genWeight");
   auto hMET_ttbar     = df_ttbar.Histo1D({"hMET_ttbar", "Missing Transverse Energy;MET [GeV];Normalized Events", 50, 0., 300.}, "PuppiMET_pt", "genWeight");

   auto hMbb_ttHH  = Mbb(df_ttHH).Histo1D(
       {"hMbb_ttHH", "H#rightarrowb#bar{b} candidate mass;m_{bb} [GeV];Normalized Events", 50, 0., 300.}, "m_bb", "genWeight");
   auto hMbb_HH    = Mbb(df_HH).Histo1D(
       {"hMbb_HH", "H#rightarrowb#bar{b} candidate mass;m_{bb} [GeV];Normalized Events", 50, 0., 300.}, "m_bb", "genWeight");
   auto hMbb_ttbar = Mbb(df_ttbar).Histo1D(
       {"hMbb_ttbar", "H#rightarrowb#bar{b} candidate mass;m_{bb} [GeV];Normalized Events", 50, 0., 300.}, "m_bb", "genWeight");

   auto hWHad_ttHH  = WHad(df_ttHH).Histo1D(
       {"hWHad_ttHH", "Hadronic W candidate mass;m_{jj} [GeV];Normalized Events", 50, 0., 200.}, "m_W_had", "genWeight");
   auto hWHad_HH    = WHad(df_HH).Histo1D(
       {"hWHad_HH", "Hadronic W candidate mass;m_{jj} [GeV];Normalized Events", 50, 0., 200.}, "m_W_had", "genWeight");
   auto hWHad_ttbar = WHad(df_ttbar).Histo1D(
       {"hWHad_ttbar", "Hadronic W candidate mass;m_{jj} [GeV];Normalized Events", 50, 0., 200.}, "m_W_had", "genWeight");

   ROOT::RDF::RunGraphs({hMuonPt_ttHH, hJetPt_ttHH, hNJets_ttHH, hNBJets_ttHH, hMET_ttHH,
                         hMuonPt_HH, hJetPt_HH, hNJets_HH, hNBJets_HH, hMET_HH,
                         hMuonPt_ttbar, hJetPt_ttbar, hNJets_ttbar, hNBJets_ttbar, hMET_ttbar,
                         hMbb_ttHH, hMbb_HH, hMbb_ttbar,
                         hWHad_ttHH, hWHad_HH, hWHad_ttbar});

   TCanvas c("c", "c", 800, 600);
   c.Print("results.pdf[");

   auto drawOverlay = [&](TH1D *h1, TH1D *h2, TH1D *h3, bool logY) {
      h1->SetLineColor(kRed + 1);   h1->SetLineWidth(1);
      h2->SetLineColor(kAzure + 1); h2->SetLineWidth(1);
      h3->SetLineColor(kGreen + 2); h3->SetLineWidth(1);

      h1->Scale(1.0 / h1->Integral());
      h2->Scale(1.0 / h2->Integral());
      h3->Scale(1.0 / h3->Integral());

      double maxY = std::max({h1->GetMaximum(), h2->GetMaximum(), h3->GetMaximum()});
      h1->SetMaximum(maxY * 1.3);
      h1->SetMinimum(logY ? 1e-5 : 0.);

      c.SetLogy(logY);
      h1->Draw("hist");
      h2->Draw("hist same");
      h3->Draw("hist same");

      TLegend leg(0.65, 0.75, 0.88, 0.88);
      leg.SetBorderSize(0);
      leg.SetTextSize(0.03);
      leg.AddEntry(h1, "ttHH", "l");
      leg.AddEntry(h2, "HH", "l");
      leg.AddEntry(h3, "ttbar", "l");
      leg.Draw();

      c.Print("results.pdf");
   };

   drawOverlay(hMuonPt_ttHH.GetPtr(), hMuonPt_HH.GetPtr(), hMuonPt_ttbar.GetPtr(), true);
   drawOverlay(hJetPt_ttHH.GetPtr(), hJetPt_HH.GetPtr(), hJetPt_ttbar.GetPtr(), true);
   drawOverlay(hNJets_ttHH.GetPtr(), hNJets_HH.GetPtr(), hNJets_ttbar.GetPtr(), false);
   drawOverlay(hNBJets_ttHH.GetPtr(), hNBJets_HH.GetPtr(), hNBJets_ttbar.GetPtr(), false);
   drawOverlay(hMET_ttHH.GetPtr(), hMET_HH.GetPtr(), hMET_ttbar.GetPtr(), true);
   drawOverlay(hMbb_ttHH.GetPtr(), hMbb_HH.GetPtr(), hMbb_ttbar.GetPtr(), false);
   drawOverlay(hWHad_ttHH.GetPtr(), hWHad_HH.GetPtr(), hWHad_ttbar.GetPtr(), false);

   c.Print("results.pdf]");

   return 0;
}
