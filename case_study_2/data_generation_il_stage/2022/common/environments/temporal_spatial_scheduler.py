#This module contains the spatial and the temporal parameters for the 1D vectorized Richards Equation
#Imports
import numpy as np
pars_MZs = np.loadtxt('../common/hydraulic_parameters_mzs.txt') #The estimated soil hydraulic parameters


def spatialVariables_1D():
    totalDepth      = 0.50    # in meters ('m')
    axialNodes      = 21
    totalNodes      = axialNodes
    return totalDepth, axialNodes, totalNodes

def temporal_discretization(_points):
    samplingTime   = 1*4*6*60*60
    samplingTimeInternal = (samplingTime/_points)
    simulationTime = 1*6*4*60*60
    internalTimeSteps = _points
    totalTimeSteps   = int(simulationTime/samplingTimeInternal)
    totalTimeSteps_daily = int(samplingTime/samplingTimeInternal)
    return samplingTime, samplingTimeInternal, internalTimeSteps, totalTimeSteps_daily, totalTimeSteps 

def temporalVariable():
    samplingTime   = 24*60*60 #Use an application rate of 2 hours
    samplingTimeInternal = 24*60*60
    simulationTime = 24*60*60
    internalTimeSteps = int(samplingTime/samplingTimeInternal)
    totTimeSteps   = int(simulationTime/samplingTime)
    return samplingTime, samplingTimeInternal, internalTimeSteps, totTimeSteps

#Hydraulic parameters of loamy soil
def loamySoil():
    soilPars={}
    soilPars['thetaR'] = 0.078
    soilPars['thetaS'] = 0.430
    soilPars['alpha']  = 3.6
    soilPars['n']      = 1.56
    soilPars['m']      = 1-1/soilPars['n']
    soilPars['Ks']     = 0.00000288889
    soilPars['neta']   = 0.5
    soilPars['Ss']     = 0.00001
    return soilPars

def pars_mz1(): # Hydraulic parameters for management zone 1
    soilPars={}
    soilPars['thetaR'] = 0.078
    soilPars['thetaS'] = 0.430
    soilPars['alpha']  = round(pars_MZs[0,2], 1)
    soilPars['n']      = 1.56
    soilPars['m']      = 1-1/round(soilPars['n'], 2)
    soilPars['Ks']     = (round(pars_MZs[0,0], 2))/86400
    soilPars['neta']   = 0.5
    soilPars['Ss']     = 0.00001
    return soilPars

def pars_mz2(): # Hydraulic parameters for management zone 2
    soilPars={}
    soilPars['thetaR'] = 0.078
    soilPars['thetaS'] = 0.430
    soilPars['alpha']  = round(pars_MZs[1,2], 1)
    soilPars['n']      = 1.56
    soilPars['m']      = 1-1/round(soilPars['n'], 2)
    soilPars['Ks']     = (round(pars_MZs[1,0], 2))/86400
    soilPars['neta']   = 0.5
    soilPars['Ss']     = 0.00001
    return soilPars

def pars_mz3(): # Hydraulic parameters for management zone 3
    soilPars={}
    soilPars['thetaR'] = 0.100
    soilPars['thetaS'] = 0.390
    soilPars['alpha']  = round(pars_MZs[2,2], 1)
    soilPars['n']      = round(pars_MZs[2,3], 2)
    soilPars['m']      = 1-1/round(soilPars['n'], 2)
    soilPars['Ks']     = (round(pars_MZs[2,0], 2))/86400
    soilPars['neta']   = 0.5
    soilPars['Ss']     = 0.00001
    return soilPars
