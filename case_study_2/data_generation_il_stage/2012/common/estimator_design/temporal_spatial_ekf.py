##The spatial and temporal variables for the 1D model(spatial variable in the axial direction)
import casadi as cs
import numpy as np
#Load the distribution of the management zones along the azimuthal nodes and the soil parameters for each management zone
pars_MZs = np.loadtxt('./common/hydraulic_parameters_mzs.txt')

def spatialVariables_1D():
    totalDepth      = 0.50   # in meters ('m')
    axialNodes      = 21
    axialDistance   = totalDepth/(axialNodes-1)
    totalNodes      = axialNodes
    # totalStates     = totalNodes
    # dimenProcNoise  = totalStates
    return totalDepth, axialNodes, axialDistance, totalNodes

def temporalVariable():
    samplingTime   = 1*4*6*60*60
    samplingTimeInternal = 1*4*6*60*60
    simulationTime = 1*4*6*60*60
    internalTimeSteps = int(samplingTime/samplingTimeInternal)
    totTimeSteps   = int(simulationTime/samplingTime)
    return samplingTime, samplingTimeInternal, internalTimeSteps, totTimeSteps

def CMatrixAllMeasurements(totalStates):
    CMatrixAll = cs.DM.eye(totalStates)
    return CMatrixAll

def loamySoil():
    soilPars={}
    soilPars['thetaR'] = 0.078
    soilPars['thetaS'] = 0.43
    soilPars['alpha']  = 3.6
    soilPars['n']      = 1.56
    soilPars['m']      = 1-1/soilPars['n']
    soilPars['Ks']     = 0.00000288889
    soilPars['neta']   = 0.5
    soilPars['Ss']     = 0.00001
    return soilPars

def obtain_soil_parameters_mz1(mz_index):
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

def obtain_soil_parameters_mz2(mz_index):
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

def obtain_soil_parameters_mz3(mz_index):
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


def generatingCoordinates_1D(axialNodes):
    #Coordinates are used to identify each node present in the mesh
    coordinates=[]
    for k in range(axialNodes):
        y=[0,0,k]
        coordinates.append(y)            
    return coordinates