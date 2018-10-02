
# coding: utf-8

# In[1]:


import json
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm


# In[2]:


def distanceEta(ieta1, ieta2):
    deta = 0
    if ieta1 * ieta2 > 0:  # Okay, they are on the same side of eta
        deta = ieta1 - ieta2
    else:
        deta = ieta1 - ieta2 - 1

    return deta


# In[3]:


def distancePhi(iphi1, iphi2):
    dphi = 0
    thisSign = np.sign(iphi1 - iphi2)
    dphi = thisSign * (abs(iphi1 - iphi2) % 360)
    return dphi


# In[4]:


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
        IETA = (i + 1 - IPHI) // 360
    # print(i,IETA,IPHI)
    return (IETA, IPHI)


# In[5]:


def convertEvent(thisEvent):

    centerEta = 85
    centerPhi = 100
    npEta = 0
    npPhi = 0
    reducedEvent = np.zeros((28, 28))

    # print(thisEvent)
    thisEventJSON = json.loads(thisEvent)
    if thisEventJSON is None:
        return reducedEvent
    for key in thisEventJSON.keys():
        (ieta, iphi) = convertCrystalNumberToEtaPhi(int(key), False)
        # print((ieta,iphi),thisEventJSON[key])
        deta = distanceEta(ieta, centerEta)
        dphi = distancePhi(iphi, centerPhi)
        if abs(deta) > 13 or abs(dphi) > 13:
            continue
        npEta = 14 + deta
        npPhi = 14 + dphi

        # print("deta is",deta,"dphi is",dphi)
        try:
            reducedEvent[npPhi][npEta] = thisEventJSON[key]
        except indexError:
            0

    return reducedEvent


# In[6]:


def convert(real_signal):

    filename = "eminus_Ele-Eta0-PhiPiOver2-Energy50.json"
    # First we open the file
    with open(filename, "r") as f:
        content = f.readlines()
        numEvents = len(content)
        print(numEvents)
        for i in range(0, numEvents):
            if i % 1000 == 0:
                print(i)
            thisEvent = content[i]
            reducedEvent = convertEvent(thisEvent)
            real_signal[0, :, :] += reducedEvent[:, :]


# In[7]:


real_signal = np.zeros((1, 28, 28))
convert(real_signal)


# In[8]:


real_signal_toplot = np.log1p(real_signal / 21000)
fig = plt.figure()
plt.imshow(real_signal_toplot[0, :, :])
cbar = plt.colorbar()
cbar.ax.set_ylabel("Energy [GeV]", rotation=270, fontsize=18, labelpad=25)
plt.xlabel("ϕ", fontsize=16)
plt.ylabel("η", fontsize=16)
plt.show()
