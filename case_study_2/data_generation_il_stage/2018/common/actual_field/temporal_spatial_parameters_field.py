##The spatial and temporal variables for the 1D model(spatial variable in the axial direction)
import numpy as np 
import os 
# file_path = os.path.join(os.path.dirname(__file__), "hydraulic_parameters_mzs.txt")
pars_MZs = np.loadtxt('./common/hydraulic_parameters_mzs.txt')
#Wheat was planted in this quadrant and the rooting depth was found to be between 0.5 - 1.0 m
def spatialVariables_1D_act():
    totalDepth      = 1.0    # in meters ('m')
    axialNodes      = 31
    totalNodes      = axialNodes
    return totalDepth, axialNodes, totalNodes
def temporalVariable():
    samplingTime   = 1*4*6*60*60
    samplingTimeInternal = 1*4*6*60*60
    simulationTime = 1*6*4*60*60
    internalTimeSteps = int(samplingTime/samplingTimeInternal)
    totTimeSteps   = int(simulationTime/samplingTime)
    return samplingTime, samplingTimeInternal, internalTimeSteps, totTimeSteps  

def temporal_discretization(_points):
    samplingTime   = 1*4*6*60*60
    samplingTimeInternal = (samplingTime/_points)
    simulationTime = 1*6*4*60*60
    internalTimeSteps = _points
    totalTimeSteps   = int(simulationTime/samplingTimeInternal)
    totalTimeSteps_daily = int(samplingTime/samplingTimeInternal)
    return samplingTime, samplingTimeInternal, internalTimeSteps, totalTimeSteps_daily, totalTimeSteps  

def generatingCoordinates_1D(axialNodes):
    #Coordinates are used to identify each node present in the mesh
    coordinates=[]
    for k in range(axialNodes):
        y=[0,0,k]
        coordinates.append(y)            
    return coordinates

def pars_mz1(): #Loamy soil
    soilPars={}
    soilPars['thetaR'] = 0.078
    soilPars['thetaS'] = 0.43
    soilPars['alpha']  = round(pars_MZs[0,2], 1)
    soilPars['n']      = 1.56
    soilPars['m']      = 1-1/round(soilPars['n'], 2)
    soilPars['Ks']     = (round(pars_MZs[0,0], 2))/86400
    soilPars['neta']   = 0.5
    soilPars['Ss']     = 0.00001
    return soilPars

def pars_mz2(): # Loamy soil
    soilPars={}
    soilPars['thetaR'] = 0.078
    soilPars['thetaS'] = 0.43
    soilPars['alpha']  = round(pars_MZs[1,2], 1)
    soilPars['n']      = 1.56
    soilPars['m']      = 1-1/round(soilPars['n'], 2)
    soilPars['Ks']     = (round(pars_MZs[1,0], 2))/86400
    soilPars['neta']   = 0.5
    soilPars['Ss']     = 0.00001
    return soilPars

def pars_mz3(): #Sandy clay loam
    soilPars={}
    soilPars['thetaR'] = 0.10
    soilPars['thetaS'] = 0.39 
    soilPars['alpha']  = round(pars_MZs[2,2], 1)
    soilPars['n']      = round(pars_MZs[2,3], 2)
    soilPars['m']      = 1-1/round(soilPars['n'], 2)
    soilPars['Ks']     = (round(pars_MZs[2,0], 2))/86400
    soilPars['neta']   = 0.5
    soilPars['Ss']     = 0.00001
    return soilPars
