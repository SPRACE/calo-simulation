# coding: utf-8

import sys
import json
from math import sqrt
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.colors as colors

### Basic definitions
SIZEOFREDUCEDSAMPLE = int(sys.argv[1])
CENTERETA = 86  # Depends on convention
CENTERPHI = 101  # Depends on convention


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


# Use only for tests
# for i in range(0,61200,1): print(i,convertCrystalNumberToEtaPhi(i,False))


def convertEvent(thisEvent, centerEta, centerPhi, numEta, numPhi):

    radiusEta = (numEta - 1) // 2
    radiusPhi = (numPhi - 1) // 2
    npEta = 0
    npPhi = 0

    reducedEvent = np.zeros((numEta, numPhi))

    # print(thisEvent)
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


def convert(filename, signal, centerEta, centerPhi):

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


signal = np.zeros((SIZEOFREDUCEDSAMPLE, SIZEOFREDUCEDSAMPLE))
listOfSignals = convert(
    "eminus_Ele-Eta0-PhiPiOver2-Energy50.json", signal, CENTERETA, CENTERPHI
)

arrayOfSignals = np.array(listOfSignals)
arrayOfSignals.shape

np.save("test.npy", arrayOfSignals, allow_pickle=False)

# # Daqui pra baixo é só validação e plotagem

#arrayOfSignals.sum(axis=0) / arrayOfSignals.shape[0]  # average
#fig = plt.figure()
#plt.imshow(signalt[:, :], norm=colors.LogNorm(), vmin=1E-2, vmax=10)
#cbar = plt.colorbar()
#cbar.ax.set_ylabel("Energy [GeV]", rotation=270, fontsize=18, labelpad=25)
#plt.xlabel("ϕ", fontsize=16)
#plt.ylabel("η", fontsize=16)
#plt.show()

sumOfEnergies = []
for s in range(0, arrayOfSignals.shape[0]):
    sumOfEnergy = arrayOfSignals[s].sum(axis=0).sum(axis=0)
    # print(sumOfEnergy)
    sumOfEnergies.append(sumOfEnergy)
print(len(sumOfEnergies))

print(
    "Mean energy =",
    np.mean(sumOfEnergies),
    "+-",
    np.std(sumOfEnergies) / sqrt(len(sumOfEnergies)),
)
