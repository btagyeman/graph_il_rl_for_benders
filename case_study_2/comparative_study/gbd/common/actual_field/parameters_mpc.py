#This module contains the spatial, temporal, and soil hydraulic parameters
import numpy as np 
pars_MZs = np.loadtxt('../common/hydraulic_parameters_mzs.txt')
def spatialVariables_1D():
    totalDepth      = 0.32   # in meters ('m')
    axialNodes      = 17
    axialDistance   = totalDepth/(axialNodes-1)
    totalNodes      = axialNodes
    numOfActuators  = 1
    interPoints     = 500
    return totalDepth, axialNodes, axialDistance, totalNodes, numOfActuators, interPoints

def temporalVariable():
    samplingTime   = 1*6*4*60*60
    samplingTimeInternal = 1*6*4*60*60
    simulationTime = 1*6*4*60*60
    internalTimeSteps = int(samplingTime/samplingTimeInternal)
    totTimeSteps   = int(simulationTime/samplingTime)
    return samplingTime, samplingTimeInternal, internalTimeSteps, totTimeSteps

def ManagementZone_1(): # Loamy soil
    soilPars = {}
    soilPars['thetaR'] = 0.078
    soilPars['thetaS'] = 0.43
    soilPars['alpha']  = round(pars_MZs[0,2], 1)
    #soilPars['n']      = pars_MZs[0,3]
    soilPars['n']      = 1.56
    soilPars['m']      = 1-1/soilPars['n']
    soilPars['Ks']     = (round(pars_MZs[0,0], 2))/86400
    soilPars['neta']   = 0.5
    soilPars['Ss']     = 0.00001
    return soilPars

def ManagementZone_2(): #Loamy soil
    soilPars = {}
    soilPars['thetaR'] = 0.078
    soilPars['thetaS'] = 0.43
    soilPars['alpha']  = round(pars_MZs[1,2],1)
    #soilPars['n']      = pars_MZs[1,3]
    soilPars['n']      = 1.56
    soilPars['m']      = 1-1/soilPars['n']
    soilPars['Ks']     = (round(pars_MZs[1,0],2))/86400
    soilPars['neta']   = 0.5
    soilPars['Ss']     = 0.00001
    return soilPars

def ManagementZone_3(): #Sandy clay loam soil
    soilPars = {}
    soilPars['thetaR'] = 0.10
    soilPars['thetaS'] = 0.39
    soilPars['alpha']  = round(pars_MZs[2,2], 1)
    soilPars['n']      = round(pars_MZs[2,3], 2)
    soilPars['m']      = 1-1/soilPars['n']
    soilPars['Ks']     = (round(pars_MZs[2,0], 2))/86400
    soilPars['neta']   = 0.5
    soilPars['Ss']     = 0.00001
    return soilPars
