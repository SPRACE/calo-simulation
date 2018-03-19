// -*- C++ -*-
//
// Package:    Subsystem/SimHitsDumper
// Class:      SimHitsDumper
// 
/**\class SimHitsDumper SimHitsDumper.cc Subsystem/SimHitsDumper/plugins/SimHitsDumper.cc

 Description: [one line class summary]

 Implementation:
     [Notes on implementation]
*/
//
// Original Author:  Thiago Tomei Fernandez
//         Created:  Tue, 13 Mar 2018 18:32:43 GMT
//
//


// system include files
#include <memory>
#include <iostream>
#include <fstream>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/DetId/interface/DetIdCollection.h"
#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "SimDataFormats/CaloHit/interface/PCaloHitContainer.h"

//
// class declaration
//

// If the analyzer does not use TFileService, please remove
// the template argument to the base class so the class inherits
// from  edm::one::EDAnalyzer<> and also remove the line from
// constructor "usesResource("TFileService");"
// This will improve performance in multithreaded jobs.

class SimHitsDumper : public edm::one::EDAnalyzer<>  {
   public:
      explicit SimHitsDumper(const edm::ParameterSet&);
      ~SimHitsDumper();

      static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);


   private:
      virtual void beginJob() override;
      virtual void analyze(const edm::Event&, const edm::EventSetup&) override;
      virtual void endJob() override;
  const size_t numCrystals = 6*360;
  const size_t offsetIndex = 29520;
  std::vector<double> crystalMap;
  edm::EDGetTokenT<edm::PCaloHitContainer> theCaloHitsToken;
  std::ofstream myfile;
      // ----------member data ---------------------------
};

//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
SimHitsDumper::SimHitsDumper(const edm::ParameterSet& iConfig)

{
   //now do what ever initialization is needed
    theCaloHitsToken = consumes<edm::PCaloHitContainer>(iConfig.getParameter<edm::InputTag>("theCaloHits"));
    crystalMap.reserve(numCrystals);
    crystalMap.resize(numCrystals);
    myfile.open("output.txt");
}


SimHitsDumper::~SimHitsDumper()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called for each event  ------------
void
SimHitsDumper::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   using namespace edm;
   
   std::fill(crystalMap.begin(),crystalMap.end(),0.0);

   // vector<PCaloHit> "g4SimHits"  "EcalHitsEB"  "SIM"
   Handle<edm::PCaloHitContainer> pCaloHits;
   iEvent.getByToken(theCaloHitsToken,pCaloHits);

   /// What are the minimum/maximum dense indexes that I want?
   /// I want ieta from -3 to 3, iphi from 1 to 360
   // std::cout << EBDetId(-3, 1).numberByEtaPhi() << std::endl;
   // std::cout << EBDetId(-3, 360).numberByEtaPhi() << std::endl;
   // std::cout << EBDetId(-2, 1).numberByEtaPhi() << std::endl;
   // std::cout << EBDetId(-1, 360).numberByEtaPhi() << std::endl;
   // std::cout << EBDetId(1, 1).numberByEtaPhi() << std::endl;
   // std::cout << EBDetId(3, 1).numberByEtaPhi() << std::endl;
   // std::cout << EBDetId(3, 360).numberByEtaPhi() << std::endl;
   //29520 29879 29880 30599 30600 31320 31679
   std::cout << "Found " << pCaloHits->size() << " CaloHits" << std::endl;

   size_t maxCaloHit = 0;
   double maxEnergy = 0;
   for(size_t i = 0; i != pCaloHits->size(); ++i) {
     const PCaloHit& ch = pCaloHits->at(i);
     double hitEnergy = ch.energy();
     if(hitEnergy > maxEnergy) {
       maxEnergy = hitEnergy;
       maxCaloHit = i;
     }
     int32_t rawid = ch.id();
     EBDetId ebDetId(rawid);
     size_t denseIndex = ebDetId.numberByEtaPhi();
     if(denseIndex >= offsetIndex and denseIndex < (offsetIndex + numCrystals)) {
       size_t convIndex = denseIndex - offsetIndex;
       crystalMap.at(convIndex) += hitEnergy;
     }
            /// Maximum caloHit
     const PCaloHit& mch = pCaloHits->at(maxCaloHit);
     int32_t maxRawid = mch.id();
     EBDetId maxEbDetId(maxRawid);
   }
   
   std::cout << "Total energy = " << std::accumulate(crystalMap.begin(), crystalMap.end(), 0.0) << std::endl;
   for(auto i = crystalMap.begin(); i != crystalMap.end(); ++i) {
     myfile << *i << '\t';
   }
   myfile << std::endl;
}



// ------------ method called once each job just before starting event loop  ------------
void 
SimHitsDumper::beginJob()
{
}

// ------------ method called once each job just after ending the event loop  ------------
void 
SimHitsDumper::endJob() 
{
  myfile.close();  
}

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void
SimHitsDumper::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  //The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  edm::ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault(desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(SimHitsDumper);
