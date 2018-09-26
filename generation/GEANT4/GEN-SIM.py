# Auto generated configuration file
# using:
# Revision: 1.19
# Source: /local/reps/CMSSW/CMSSW/Configuration/Applications/python/ConfigBuilder.py,v
# with command line options: Configuration/GenProduction/python/SingleElectronFlatE20To100_pythia8_cfi.py --fileout file:GEN-SIM.root -s GEN,SIM --mc --datatier GEN-SIM --beamspot Realistic25ns13TeV2016Collision --conditions auto:phase1_2017_realistic --eventcontent RAWSIM --era Run2_2017 --python_filename GEN-SIM.py --no_exec -n 10
import FWCore.ParameterSet.Config as cms

from Configuration.StandardSequences.Eras import eras
import random

process = cms.Process("SIM", eras.Run2_2017)

# import of standard configurations
process.load("Configuration.StandardSequences.Services_cff")
process.load("SimGeneral.HepPDTESSource.pythiapdt_cfi")
process.load("FWCore.MessageService.MessageLogger_cfi")
process.load("Configuration.EventContent.EventContent_cff")
process.load("SimGeneral.MixingModule.mixNoPU_cfi")
process.load("Configuration.StandardSequences.GeometryRecoDB_cff")
process.load("Configuration.Geometry.GeometrySimDB_cff")
process.load("Configuration.StandardSequences.MagneticField_cff")
process.load("Configuration.StandardSequences.Generator_cff")
process.load("IOMC.EventVertexGenerators.VtxSmearedRealistic25ns13TeV2016Collision_cfi")
process.load("GeneratorInterface.Core.genFilterSummary_cff")
process.load("Configuration.StandardSequences.SimIdeal_cff")
process.load("Configuration.StandardSequences.EndOfProcess_cff")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")

process.maxEvents = cms.untracked.PSet(input=cms.untracked.int32(1000))

rn1 = random.randint(1, 10000000)
rn2 = random.randint(1, 10000000)
process.RandomNumberGeneratorService.generator.initialSeed = rn1
process.RandomNumberGeneratorService.g4SimHits.initialSeed = rn2

# Input source
process.source = cms.Source("EmptySource")

process.options = cms.untracked.PSet()

# Production Info
process.configurationMetadata = cms.untracked.PSet(
    annotation=cms.untracked.string(
        "Configuration/GenProduction/python/SingleElectronFlatE20To100_pythia8_cfi.py nevts:10"
    ),
    name=cms.untracked.string("Applications"),
    version=cms.untracked.string("$Revision: 1.19 $"),
)

# Output definition

process.RAWSIMoutput = cms.OutputModule(
    "PoolOutputModule",
    SelectEvents=cms.untracked.PSet(SelectEvents=cms.vstring("generation_step")),
    compressionAlgorithm=cms.untracked.string("LZMA"),
    compressionLevel=cms.untracked.int32(9),
    dataset=cms.untracked.PSet(
        dataTier=cms.untracked.string("GEN-SIM"), filterName=cms.untracked.string("")
    ),
    eventAutoFlushCompressedSize=cms.untracked.int32(20971520),
    fileName=cms.untracked.string(
        "file:GEN-SIM_Electron-Eta0-PhiPiOver2-Energy20_" + str(rn2) + ".root"
    ),
    outputCommands=process.RAWSIMEventContent.outputCommands,
    splitLevel=cms.untracked.int32(0),
)

# Additional output definition

# Other statements
process.genstepfilter.triggerConditions = cms.vstring("generation_step")
from Configuration.AlCa.GlobalTag import GlobalTag

process.GlobalTag = GlobalTag(process.GlobalTag, "auto:phase1_2017_realistic", "")

process.generator = cms.EDFilter(
    "Pythia8EGun",
    PGunParameters=cms.PSet(
        AddAntiParticle=cms.bool(False),
        MaxE=cms.double(20.000001),
        MinE=cms.double(19.999999),
        MaxEta=cms.double(1e-11),
        MinEta=cms.double(-1e-11),
        MaxPhi=cms.double(1.570796327 + 1e-11),
        MinPhi=cms.double(1.570796327 - 1e-11),
        ParticleID=cms.vint32(13),
    ),
    PythiaParameters=cms.PSet(parameterSets=cms.vstring()),
    Verbosity=cms.untracked.int32(0),
    firstRun=cms.untracked.uint32(1),
    psethack=cms.string("single electron energy 20"),
)


# Path and EndPath definitions
process.generation_step = cms.Path(process.pgen)
process.simulation_step = cms.Path(process.psim)
process.genfiltersummary_step = cms.EndPath(process.genFilterSummary)
process.endjob_step = cms.EndPath(process.endOfProcess)
process.RAWSIMoutput_step = cms.EndPath(process.RAWSIMoutput)

# Schedule definition
process.schedule = cms.Schedule(
    process.generation_step,
    process.genfiltersummary_step,
    process.simulation_step,
    process.endjob_step,
    process.RAWSIMoutput_step,
)
from PhysicsTools.PatAlgos.tools.helpers import associatePatAlgosToolsTask

associatePatAlgosToolsTask(process)
# filter all path with the production filter sequence
for path in process.paths:
    getattr(process, path)._seq = process.generator * getattr(process, path)._seq


# Customisation from command line

# Add early deletion of temporary data products to reduce peak memory need
from Configuration.StandardSequences.earlyDeleteSettings_cff import customiseEarlyDelete

process = customiseEarlyDelete(process)
# End adding early deletion
