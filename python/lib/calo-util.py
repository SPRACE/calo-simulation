
# coding: utf-8

import json 
import numpy as np 

import time

from math import sqrt

import matplotlib.pyplot as plt 
import matplotlib.colors as colors

import pandas as pd 

from scipy import stats

import plotly
import plotly.plotly as py
from plotly.offline import iplot
import plotly.graph_objs as go
plotly.offline.init_notebook_mode()

import os

### Basic definitions
DEFSIZEOFREDUCEDSAMPLE = 28
DEFCENTERETA           = 86   # Depends on convention
DEFCENTERPHI           = 101  # Depends on convention
DEFRANGEPHI            = 360
DEFRANGEETA            = 170
RANGETOTAL             = DEFRANGEPHI * DEFRANGEETA


###
### Json -> npy 
###


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


def convert(filename, signal, centerEta = DEFCENTERETA, centerPhi = DEFCENTERPHI):
    '''
    Convert json file default values for 
    SIZEOFREDUCEDSAMPLE, CENTERETA, CENTERPHI 
    and return a list of signals 
    '''
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


timestr = time.strftime("%Y%m%d%H%M%S")
defaultFileName = "signals_" + timestr + ".npy"

def generate(inputFilename, outputFilename = defaultFileName, sizeOfReducedSample = DEFSIZEOFREDUCEDSAMPLE, centerEta = DEFCENTERETA, centerPhi = DEFCENTERPHI):
    '''
    Convert json file default values for 
    SIZEOFREDUCEDSAMPLE, CENTERETA, CENTERPHI 
    and return a list of signals 
    '''
    
    signal = np.zeros((sizeOfReducedSample, sizeOfReducedSample))
    listOfSignals = convert(
        filename, signal, centerEta, centerPhi
    )

    arrayOfSignals = np.array(listOfSignals)
    arrayOfSignals.shape

    np.save(outputFilename, arrayOfSignals, allow_pickle=False)

### END
###
### Json -> npy 
###


###
### Charts
###


def plotLines(data, name, xtitle, ytitle, mode):
    data_points = go.Scatter(
        x = data,
        mode = 'lines',
        name = 'lines'
    )

    plot_data = [data_points]
    
    if mode=="x_log":
        layout = go.Layout(
            title=name,
            titlefont = dict(
            size = 14
            ),
            xaxis=dict(
                title= "log(" + xtitle + ")",
                type='log',
                autorange=True
            ),
            yaxis=dict(
                title=ytitle,
                autorange=True
            ),

        )
    elif mode=="y_log":
        layout = go.Layout(
            title=name,
            titlefont = dict(
            size = 14
            ),
            xaxis=dict(
                title=xtitle,
                autorange=True
            ),
            yaxis=dict(
                title="log(" + ytitle + ")",
                type='log',
                autorange=True
            ),

        )
    elif mode=="xy_log":
        layout = go.Layout(
            title=name,
            titlefont = dict(
            size = 14
            ),
            xaxis=dict(
                title="log(" + xtitle + ")",
                type='log',
                autorange=True
            ),
            yaxis=dict(
                title="log(" + ytitle + ")",
                type='log',
                autorange=True
            ),

        )
    else:
        layout = go.Layout(
            title=name,
            titlefont = dict(
            size = 14
            ),
            xaxis=dict(
                title=xtitle,
                autorange=True
            ),
            yaxis=dict(
                title=ytitle,
                autorange=True
            ),

        )
        
    fig = go.Figure(data=plot_data, layout=layout)
    
    iplot(fig, filename="jupyter/{}".format(name))


def sttstcs(path):
    data = []
    total_lines = 0
    with open(path) as file:
        for line in file:
            data.append(json.loads(line))
            total_lines+=1
    signal = np.zeros(RANGETOTAL)
    signal_avg = np.zeros(RANGETOTAL)
    
    statistics_signal = {}
    #statistics_signal_ = {}

    valid = np.empty(total_lines)
    valid[:] = -1

    signals = {}
    statistics_signals = {}

    for i in range(total_lines):
        try:
            keys = (np.asarray((list(data[i].keys())))).astype(int)
            values = np.asarray(list(data[i].values()))
            signals[i] = [keys,values,]
            statistics_signals[i] = stats.describe(values)
            valid[i] = 1 #valid signals 
            signal[keys] += values #sum of all signals 
        except: 
            continue 
    
    energy_per_signal = np.zeros(total_lines)-1
    
    for i in range(total_lines):
        try:
            energy_per_signal[i] = np.asarray(signals[i][1]).sum()
        except:
            continue       
    
    signal_energy = signal.sum()/len(valid[valid!=-1])
    
    S = signal.reshape(DEFRANGEETA,DEFRANGEPHI)
    
    signal_avg = signal/len(valid[valid!=-1])
    signal_avg = signal_avg.reshape(DEFRANGEETA,DEFRANGEPHI)
    
    #X var (0 - 360)
    VarX  = S.var(axis=0)
    MeanX = S.mean(axis=0)
    MaxX  = S.max(axis=0)
    MinX  = S.min(axis=0)
    
    #Y var (0 - 170)
    VarY  = S.var(axis=1)
    MeanY = S.mean(axis=1)
    MaxY  = S.max(axis=1)
    MinY  = S.min(axis=1)
    
    
    
    proj_phi = signal_avg.sum(axis=0)
    proj_eta = signal_avg.sum(axis=1)
    
    return signal_avg
    

def projectionPhi(proj_phi):
    return proj_phi.sum(axis=0)


def projectionEta(proj_eta):
    return proj_eta.sum(axis=1)


### END
###
### Charts
###





