import FWCore.ParameterSet.Config as cms

process = cms.Process("Demo")

process.load("FWCore.MessageService.MessageLogger_cfi")

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(-1) )

process.source = cms.Source("PoolSource",
    # replace 'myfile.root' with the source file you want to use
    fileNames = cms.untracked.vstring(
        'file:GEN-SIM.root'
    )
)

process.load("SimGeneral.HepPDTESSource.pythiapdt_cfi")
process.printTree = cms.EDAnalyzer("ParticleListDrawer",
                                   maxEventsToPrint = cms.untracked.int32(100),
                                   printVertex = cms.untracked.bool(False),
                                   printOnlyHardInteraction = cms.untracked.bool(False),
                                   src = cms.InputTag("genParticles")
                                   #src = cms.InputTag("gedGsfElectrons")
)

process.demo = cms.EDAnalyzer('SimHitsDumper',
                              theCaloHits = cms.InputTag("g4SimHits","EcalHitsEB")
                              )


process.p = cms.Path(process.printTree + process.demo)
