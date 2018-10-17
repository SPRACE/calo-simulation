import json
from math import sqrt
import numpy as np
import os
import time

### Basic definitions
DEFSIZEOFREDUCEDSAMPLE = 28
DEFCENTERETA           = 86   # Depends on convention
DEFCENTERPHI           = 101  # Depends on convention
DEFRANGEPHI            = 360
DEFRANGEETA            = 170
RANGETOTAL             = DEFRANGEPHI * DEFRANGEETA


def mean_component(data, axis):
    component_sum = np.sum(data, axis=axis)
    component_mean = np.mean(component_sum, axis=0)
    return component_mean
    
def mean_eta(data):
    return mean_component(data, axis=1)

def mean_phi(data):
    return mean_component(data, axis=2)
               
    
def sum_energy(data):
    if len(data.shape) == 2:
        return np.sum(data, axis=(0,1))
    else:
        return np.sum(data, axis=(1,2))


###
### Json -> npy 
###

def sampleArray(sizeOfReducedSample = DEFSIZEOFREDUCEDSAMPLE):
    """Function to return a array of zeroes with default value of reduced 
    sample dimension. 

    Args:
        sizeOfReducedSample (integer): value of reduced sample dimension.

    Returns:
        numpy.ndarray: array of zeroes.

    """    
    return np.zeros((sizeOfReducedSample, sizeOfReducedSample))
    

def distanceEta(ieta1, ieta2):
    # FIXME: make this function work with CMS convention
    deta = 0
    if ieta1 * ieta2 > 0:  # Okay, they are on the same side of eta
        deta = ieta1 - ieta2
    else:
        deta = ieta1 - ieta2 - 1
    return deta


def distancePhi(iphi1, iphi2):
    # FIXME: make this function work with CMS convention
    dphi = 0
    thisSign = np.sign(iphi1 - iphi2)
    dphi = thisSign * (abs(iphi1 - iphi2) % 360)
    return dphi


def convertCrystalNumberToEtaPhi(CRYSTALNUMBER, CMSConvention):
    """
    CRYSTALNUMBER goes from 0 to 61199
    IETA goes from -85 to +85, and there is no zero
    IPHI goes from 1 to 360
    """
    i = CRYSTALNUMBER
    IPHI = i % 360 + 1
    IETA = 0
    SIGN = 0

    # Check SIGN
    if i > 30599:
        SIGN = 1
    if not (i > 30599):
        SIGN = -1

    # Correct ETA
    if CMSConvention is True:
        if SIGN > 0:
            IETA = (i + 1 - IPHI) // 360 - 84
        if SIGN < 0:
            IETA = (i + 1 - IPHI) // 360 - 85
    else:
        """
        If we don't follow the CMS convention,
        IETA goes from 1 to 170
        IPHI goes from 1 to 360
        """
        IETA = (i - IPHI) // 360 + 2
    # print(i,IETA,IPHI)
    return (IETA, IPHI)


def convertEvent(thisEvent, centerEta, centerPhi, numEta, numPhi):
    radiusEta = (numEta - 1) // 2
    radiusPhi = (numPhi - 1) // 2
    npEta = 0
    npPhi = 0
    reducedEvent = np.zeros((numEta, numPhi))
    thisEventJSON = json.loads(thisEvent)
    if thisEventJSON is None:
        return reducedEvent
    for key in thisEventJSON.keys():
        (ieta, iphi) = convertCrystalNumberToEtaPhi(int(key), False)
        # print((ieta,iphi),thisEventJSON[key])
        deta = distanceEta(ieta, centerEta)
        dphi = distancePhi(iphi, centerPhi)
        if abs(deta) > radiusEta or abs(dphi) > radiusPhi:
            continue
        npEta = radiusEta + deta
        npPhi = radiusPhi + dphi
        # print("deta is",deta,"dphi is",dphi)
        reducedEvent[npEta][npPhi] = thisEventJSON[key]
    return reducedEvent


def convert(filename, signal=sampleArray(), 
    centerEta = DEFCENTERETA, centerPhi = DEFCENTERPHI):
    """Function to convert json containing the energy deposition 
                                       at the calorimeter cells in eta or phi
                                       in a numpy array.
    Args:
        filename (string): A json path containing the energy deposition 
                                       at the calorimeter cells in eta or phi.
        signal (numpy.ndarray): A array of zeroes with value of reduced 
                                       sample dimension (DEFAULT:sampleArray()).
        centerEta (integer): The value of center Eta (DEFAULT:DEFCENTERETA). 
        centerPhi (integer): The value of center Phi (DEFAULT:DEFCENTERPHI).

    Returns:
        numpy.ndarray: Array of conversion Json -> npy.

    """    
    numEvents = 0
    listOfSignals = []
    # First we open the file
    with open(filename, "r") as f:
        content = f.readlines()
        numEvents = len(content)
        for i in range(0, numEvents):
            if i % 1000 == 0:
                print(i)
            thisEvent = content[i]
            try:
                reducedEvent = convertEvent(
                    thisEvent, centerEta, centerPhi, signal.shape[0], signal.shape[1]
                )                                                                                                                                                                                                                                                                                                          
                listOfSignals.append(reducedEvent)
            except indexError:
                0
    print("Converted", len(listOfSignals), "out of", numEvents, "events")
    return listOfSignals

#Generating .npy filename using datetime
timestr = time.strftime("%Y%m%d%H%M%S")
defaultFileName = "signals_" + timestr + ".npy"

def generate(inputFilename, outputFilename = defaultFileName, 
    sizeOfReducedSample = DEFSIZEOFREDUCEDSAMPLE, 
    centerEta = DEFCENTERETA, centerPhi = DEFCENTERPHI):
    """Function to convert json containing the energy deposition 
                                       at the calorimeter cells in eta or phi
                                       in a numpy file (.npy).
    Args:
        inputFilename (string): A json path containing the energy deposition 
                                       at the calorimeter cells in eta or phi.
        outputFilename (string): a path of a output file (.npy). By default, 
                                       this function will be use datetime + 
                                       extension (e.g. 
                                       signals_20181017154439.npy).       
        sizeOfReducedSample (integer): The size of reduced sample
                                       (DEFAULT: DEFSIZEOFREDUCEDSAMPLE). 
        centerEta (integer): The value of center Eta (DEFAULT:DEFCENTERETA). 
        centerPhi (integer): The value of center Phi (DEFAULT:DEFCENTERPHI).

    Returns:
        .npy file: Array of conversion Json -> npy.
    """        
    listOfSignals = convert(inputFilename)
    arrayOfSignals = np.array(listOfSignals)
    arrayOfSignals.shape
    np.save(outputFilename, arrayOfSignals, allow_pickle=False)
    print("npy array name: ",outputFilename)


### END
###
### Json -> npy 
###

