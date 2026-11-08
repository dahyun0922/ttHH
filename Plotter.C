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


#include <fstream>
#include <map>
#include <stdexcept>
#include <string>

using namespace ROOT;
using namespace ROOT::VecOps;

#include "Reconstruction.h"
#include "AnalysisFeatures.h"
#include "DatasetConfig.h"

int Plotter(const char *configFile = "config/datasets.txt") {
    auto paths = ReadDatasetPaths(configFile);

   ROOT::EnableImplicitMT();
   gStyle->SetOptStat(0);

    std::string dir_ttHH = paths.at("ttHH");

    std::string dir_HH = paths.at("HH");

    std::string dir_ttbar = paths.at("ttbar");

   auto df_ttHH  = AddBJetVars(ROOT::RDataFrame("Events", dir_ttHH));
   auto df_HH    = AddBJetVars(ROOT::RDataFrame("Events", dir_HH));
   auto df_ttbar = AddBJetVars(ROOT::RDataFrame("Events", dir_ttbar));

   auto hMuonPt_ttHH  = df_ttHH.Histo1D({"hMuonPt_ttHH", "Muon p_{T};Muon p_{T} [GeV];Normalized Events", 50, 0., 200.}, "selectedMuon_pt", "genWeight");
   auto hJetPt_ttHH   = df_ttHH.Histo1D({"hJetPt_ttHH", "Jet p_{T};Jet p_{T} [GeV];Normalized Events", 50, 0., 300.}, "selectedJet_pt", "genWeight");
   auto hNJets_ttHH   = df_ttHH.Histo1D({"hNJets_ttHH", "Number of Jets;No. Jets;Normalized Events", 20, -0.5, 19.5}, "nSelectedJet", "genWeight");
   auto hNBJets_ttHH  = df_ttHH.Histo1D({"hNBJets_ttHH", "Number of b-Jets;No. b-Jets;Normalized Events", 6, -0.5, 5.5}, "nBJets", "genWeight");
   auto hMET_ttHH     = df_ttHH.Histo1D({"hMET_ttHH", "Missing Transverse Energy;MET [GeV];Normalized Events", 50, 0., 300.}, "PuppiMET_pt", "genWeight");

   auto hMuonPt_HH  = df_HH.Histo1D({"hMuonPt_HH", "Muon p_{T};Muon p_{T} [GeV];Normalized Events", 50, 0., 200.}, "selectedMuon_pt", "genWeight");
   auto hJetPt_HH   = df_HH.Histo1D({"hJetPt_HH", "Jet p_{T};Jet p_{T} [GeV];Normalized Events", 50, 0., 300.}, "selectedJet_pt", "genWeight");
   auto hNJets_HH   = df_HH.Histo1D({"hNJets_HH", "Number of Jets;No. Jets;Normalized Events", 20, -0.5, 19.5}, "nSelectedJet", "genWeight");
   auto hNBJets_HH  = df_HH.Histo1D({"hNBJets_HH", "Number of b-Jets;No. b-Jets;Normalized Events", 6, -0.5, 5.5}, "nBJets", "genWeight");
   auto hMET_HH     = df_HH.Histo1D({"hMET_HH", "Missing Transverse Energy;MET [GeV];Normalized Events", 50, 0., 300.}, "PuppiMET_pt", "genWeight");

   auto hMuonPt_ttbar  = df_ttbar.Histo1D({"hMuonPt_ttbar", "Muon p_{T};Muon p_{T} [GeV];Normalized Events", 50, 0., 200.}, "selectedMuon_pt", "genWeight");
   auto hJetPt_ttbar   = df_ttbar.Histo1D({"hJetPt_ttbar", "Jet p_{T};Jet p_{T} [GeV];Normalized Events", 50, 0., 300.}, "selectedJet_pt", "genWeight");
   auto hNJets_ttbar   = df_ttbar.Histo1D({"hNJets_ttbar", "Number of Jets;No. Jets;Normalized Events", 20, -0.5, 19.5}, "nSelectedJet", "genWeight");
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


   // Global event activity: HT and ST

   auto hHT_ttHH = df_ttHH.Histo1D(
       {"hHT_ttHH", "HT;HT [GeV];Normalized Events",
         60, 0., 1500.}, "HT", "genWeight");

   auto hHT_HH = df_HH.Histo1D(
       {"hHT_HH", "HT;HT [GeV];Normalized Events",
         60, 0., 1500.}, "HT", "genWeight");

   auto hHT_ttbar = df_ttbar.Histo1D(
       {"hHT_ttbar", "HT;HT [GeV];Normalized Events",
         60, 0., 1500.}, "HT", "genWeight");

   auto hST_ttHH = df_ttHH.Histo1D(
       {"hST_ttHH", "ST;ST [GeV];Normalized Events",
         60, 0., 2000.}, "ST", "genWeight");

   auto hST_HH = df_HH.Histo1D(
       {"hST_HH", "ST;ST [GeV];Normalized Events",
         60, 0., 2000.}, "ST", "genWeight");

   auto hST_ttbar = df_ttbar.Histo1D(
       {"hST_ttbar", "ST;ST [GeV];Normalized Events",
         60, 0., 2000.}, "ST", "genWeight");


   // Individual object and angular features

   auto h_leading_jet_pt_ttHH = df_ttHH.Histo1D(
       {"h_leading_jet_pt_ttHH", "leading_jet_pt;leading_jet_pt;Normalized Events",
         50, 0., 300.}, "leading_jet_pt", "genWeight");

   auto h_leading_jet_pt_HH = df_HH.Histo1D(
       {"h_leading_jet_pt_HH", "leading_jet_pt;leading_jet_pt;Normalized Events",
         50, 0., 300.}, "leading_jet_pt", "genWeight");

   auto h_leading_jet_pt_ttbar = df_ttbar.Histo1D(
       {"h_leading_jet_pt_ttbar", "leading_jet_pt;leading_jet_pt;Normalized Events",
         50, 0., 300.}, "leading_jet_pt", "genWeight");

   auto h_subleading_jet_pt_ttHH = df_ttHH.Histo1D(
       {"h_subleading_jet_pt_ttHH", "subleading_jet_pt;subleading_jet_pt;Normalized Events",
         50, 0., 300.}, "subleading_jet_pt", "genWeight");

   auto h_subleading_jet_pt_HH = df_HH.Histo1D(
       {"h_subleading_jet_pt_HH", "subleading_jet_pt;subleading_jet_pt;Normalized Events",
         50, 0., 300.}, "subleading_jet_pt", "genWeight");

   auto h_subleading_jet_pt_ttbar = df_ttbar.Histo1D(
       {"h_subleading_jet_pt_ttbar", "subleading_jet_pt;subleading_jet_pt;Normalized Events",
         50, 0., 300.}, "subleading_jet_pt", "genWeight");

   auto h_leading_lepton_pt_ttHH = df_ttHH.Histo1D(
       {"h_leading_lepton_pt_ttHH", "leading_lepton_pt;leading_lepton_pt;Normalized Events",
         50, 0., 200.}, "leading_lepton_pt", "genWeight");

   auto h_leading_lepton_pt_HH = df_HH.Histo1D(
       {"h_leading_lepton_pt_HH", "leading_lepton_pt;leading_lepton_pt;Normalized Events",
         50, 0., 200.}, "leading_lepton_pt", "genWeight");

   auto h_leading_lepton_pt_ttbar = df_ttbar.Histo1D(
       {"h_leading_lepton_pt_ttbar", "leading_lepton_pt;leading_lepton_pt;Normalized Events",
         50, 0., 200.}, "leading_lepton_pt", "genWeight");

   auto h_deltaR_bb_leading_ttHH = df_ttHH.Histo1D(
       {"h_deltaR_bb_leading_ttHH", "deltaR_bb_leading;deltaR_bb_leading;Normalized Events",
         50, 0., 6.}, "deltaR_bb_leading", "genWeight");

   auto h_deltaR_bb_leading_HH = df_HH.Histo1D(
       {"h_deltaR_bb_leading_HH", "deltaR_bb_leading;deltaR_bb_leading;Normalized Events",
         50, 0., 6.}, "deltaR_bb_leading", "genWeight");

   auto h_deltaR_bb_leading_ttbar = df_ttbar.Histo1D(
       {"h_deltaR_bb_leading_ttbar", "deltaR_bb_leading;deltaR_bb_leading;Normalized Events",
         50, 0., 6.}, "deltaR_bb_leading", "genWeight");


   // Additional bb reconstruction plots

   auto h_m_bb_leading_ttHH = df_ttHH
       .Filter("nBJets >= 2")
       .Histo1D(
           {"h_m_bb_leading_ttHH", "m_bb_leading;m_bb_leading [GeV];Normalized Events",
             50, 0., 300.0},
           "m_bb_leading", "genWeight");

   auto h_m_bb_leading_HH = df_HH
       .Filter("nBJets >= 2")
       .Histo1D(
           {"h_m_bb_leading_HH", "m_bb_leading;m_bb_leading [GeV];Normalized Events",
             50, 0., 300.0},
           "m_bb_leading", "genWeight");

   auto h_m_bb_leading_ttbar = df_ttbar
       .Filter("nBJets >= 2")
       .Histo1D(
           {"h_m_bb_leading_ttbar", "m_bb_leading;m_bb_leading [GeV];Normalized Events",
             50, 0., 300.0},
           "m_bb_leading", "genWeight");

   auto h_pt_bb_ttHH = df_ttHH
       .Filter("nBJets >= 2")
       .Histo1D(
           {"h_pt_bb_ttHH", "pt_bb;pt_bb [GeV];Normalized Events",
             50, 0., 500.0},
           "pt_bb", "genWeight");

   auto h_pt_bb_HH = df_HH
       .Filter("nBJets >= 2")
       .Histo1D(
           {"h_pt_bb_HH", "pt_bb;pt_bb [GeV];Normalized Events",
             50, 0., 500.0},
           "pt_bb", "genWeight");

   auto h_pt_bb_ttbar = df_ttbar
       .Filter("nBJets >= 2")
       .Histo1D(
           {"h_pt_bb_ttbar", "pt_bb;pt_bb [GeV];Normalized Events",
             50, 0., 500.0},
           "pt_bb", "genWeight");

   auto h_deltaR_bb_Hcand_ttHH = df_ttHH
       .Filter("nBJets >= 2")
       .Histo1D(
           {"h_deltaR_bb_Hcand_ttHH", "deltaR_bb_Hcand;deltaR_bb_Hcand;Normalized Events",
             50, 0., 6.0},
           "deltaR_bb_Hcand", "genWeight");

   auto h_deltaR_bb_Hcand_HH = df_HH
       .Filter("nBJets >= 2")
       .Histo1D(
           {"h_deltaR_bb_Hcand_HH", "deltaR_bb_Hcand;deltaR_bb_Hcand;Normalized Events",
             50, 0., 6.0},
           "deltaR_bb_Hcand", "genWeight");

   auto h_deltaR_bb_Hcand_ttbar = df_ttbar
       .Filter("nBJets >= 2")
       .Histo1D(
           {"h_deltaR_bb_Hcand_ttbar", "deltaR_bb_Hcand;deltaR_bb_Hcand;Normalized Events",
             50, 0., 6.0},
           "deltaR_bb_Hcand", "genWeight");

   ROOT::RDF::RunGraphs({hMuonPt_ttHH, hJetPt_ttHH, hNJets_ttHH, hNBJets_ttHH, hMET_ttHH,
                         hMuonPt_HH, hJetPt_HH, hNJets_HH, hNBJets_HH, hMET_HH,
                         hMuonPt_ttbar, hJetPt_ttbar, hNJets_ttbar, hNBJets_ttbar, hMET_ttbar,
                         hMbb_ttHH, hMbb_HH, hMbb_ttbar,
                         hWHad_ttHH, hWHad_HH, hWHad_ttbar,
                         hHT_ttHH, hHT_HH, hHT_ttbar,
                         hST_ttHH, hST_HH, hST_ttbar,
                         h_leading_jet_pt_ttHH,
                         h_leading_jet_pt_HH,
                         h_leading_jet_pt_ttbar,
                         h_subleading_jet_pt_ttHH,
                         h_subleading_jet_pt_HH,
                         h_subleading_jet_pt_ttbar,
                         h_leading_lepton_pt_ttHH,
                         h_leading_lepton_pt_HH,
                         h_leading_lepton_pt_ttbar,
                         h_deltaR_bb_leading_ttHH,
                         h_deltaR_bb_leading_HH,
                         h_deltaR_bb_leading_ttbar,
                         h_m_bb_leading_ttHH,
                         h_m_bb_leading_HH,
                         h_m_bb_leading_ttbar,
                         h_pt_bb_ttHH,
                         h_pt_bb_HH,
                         h_pt_bb_ttbar,
                         h_deltaR_bb_Hcand_ttHH,
                         h_deltaR_bb_Hcand_HH,
                         h_deltaR_bb_Hcand_ttbar});

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

   drawOverlay(hHT_ttHH.GetPtr(), hHT_HH.GetPtr(),
               hHT_ttbar.GetPtr(), true);
   drawOverlay(hST_ttHH.GetPtr(), hST_HH.GetPtr(),
               hST_ttbar.GetPtr(), true);


   drawOverlay(
       h_leading_jet_pt_ttHH.GetPtr(),
       h_leading_jet_pt_HH.GetPtr(),
       h_leading_jet_pt_ttbar.GetPtr(), false);

   drawOverlay(
       h_subleading_jet_pt_ttHH.GetPtr(),
       h_subleading_jet_pt_HH.GetPtr(),
       h_subleading_jet_pt_ttbar.GetPtr(), false);

   drawOverlay(
       h_leading_lepton_pt_ttHH.GetPtr(),
       h_leading_lepton_pt_HH.GetPtr(),
       h_leading_lepton_pt_ttbar.GetPtr(), false);

   drawOverlay(
       h_deltaR_bb_leading_ttHH.GetPtr(),
       h_deltaR_bb_leading_HH.GetPtr(),
       h_deltaR_bb_leading_ttbar.GetPtr(), false);


   drawOverlay(
       h_m_bb_leading_ttHH.GetPtr(),
       h_m_bb_leading_HH.GetPtr(),
       h_m_bb_leading_ttbar.GetPtr(), false);

   drawOverlay(
       h_pt_bb_ttHH.GetPtr(),
       h_pt_bb_HH.GetPtr(),
       h_pt_bb_ttbar.GetPtr(), false);

   drawOverlay(
       h_deltaR_bb_Hcand_ttHH.GetPtr(),
       h_deltaR_bb_Hcand_HH.GetPtr(),
       h_deltaR_bb_Hcand_ttbar.GetPtr(), false);

   c.Print("results.pdf]");

   return 0;
}
