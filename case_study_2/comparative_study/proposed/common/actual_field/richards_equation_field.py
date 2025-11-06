import numpy as np
from common.actual_field.temporal_spatial_parameters_field import *
import common.actual_field.van_Genuchten_relations as vg
totalDepth, axialNodes, totalNodes = spatialVariables_1D_act() 
nodal_distances = np.zeros(totalNodes-1)
daily_timesteps = 4
for i in range(10, totalNodes-1):
    nodal_distances[i] = 0.025
    pass
for j in range(0, 10):
    nodal_distances[j] = 0.050
    pass
approx_distances = np.zeros(totalNodes)
for k in range(0, 10):
    approx_distances[k] = 0.050
    pass
for l in range(11,totalNodes):
    approx_distances[l] = 0.025
    pass
approx_distances[10] = 0.5*(0.025 + 0.050)

actual_distances_1m = np.zeros(totalNodes)
actual_distances_1m[0]  =  0.000
actual_distances_1m[1]  =  0.050
actual_distances_1m[2]  =  0.100
actual_distances_1m[3]  =  0.150
actual_distances_1m[4]  =  0.200
actual_distances_1m[5]  =  0.250
actual_distances_1m[6]  =  0.300
actual_distances_1m[7]  =  0.350
actual_distances_1m[8]  =  0.400
actual_distances_1m[9]  =  0.450
actual_distances_1m[10] =  0.500
actual_distances_1m[11] =  0.525
actual_distances_1m[12] =  0.550
actual_distances_1m[13] =  0.575
actual_distances_1m[14] =  0.600
actual_distances_1m[15] =  0.625
actual_distances_1m[16] =  0.650
actual_distances_1m[17] =  0.675
actual_distances_1m[18] =  0.700
actual_distances_1m[19] =  0.725
actual_distances_1m[20] =  0.750
actual_distances_1m[21] =  0.775
actual_distances_1m[22] =  0.800
actual_distances_1m[23] =  0.825
actual_distances_1m[24] =  0.850
actual_distances_1m[25] =  0.875
actual_distances_1m[26] =  0.900
actual_distances_1m[27] =  0.925
actual_distances_1m[28] =  0.950
actual_distances_1m[29] =  0.975
actual_distances_1m[30] =  1.000

actual_distances_50cm = np.zeros(totalNodes)
actual_distances_50cm[10] =  0.000
actual_distances_50cm[11] =  0.025
actual_distances_50cm[12] =  0.050
actual_distances_50cm[13] =  0.075
actual_distances_50cm[14] =  0.100
actual_distances_50cm[15] =  0.125
actual_distances_50cm[16] =  0.150
actual_distances_50cm[17] =  0.175
actual_distances_50cm[18] =  0.200
actual_distances_50cm[19] =  0.225
actual_distances_50cm[20] =  0.250
actual_distances_50cm[21] =  0.275
actual_distances_50cm[22] =  0.300
actual_distances_50cm[23] =  0.325
actual_distances_50cm[24] =  0.350
actual_distances_50cm[25] =  0.375
actual_distances_50cm[26] =  0.400
actual_distances_50cm[27] =  0.425
actual_distances_50cm[28] =  0.450
actual_distances_50cm[29] =  0.475 
actual_distances_50cm[30] =  0.500


def evaluate_sink_terms(refEvap, cropCoeff, rooting_depth, lai_factor):
    sinkTerms = np.zeros(totalNodes)    
    if rooting_depth == 1.00:
        sinkTerms[:] = (2*(lai_factor*refEvap*cropCoeff)/rooting_depth)*(1 - ((rooting_depth - actual_distances_1m)/rooting_depth))
    else:
        sinkTerms[10:] = (2*(lai_factor*refEvap*cropCoeff)/rooting_depth)*(1 - ((rooting_depth - actual_distances_50cm[10:])/rooting_depth))
    return sinkTerms

def determine_surface_flux(irrigAmount, lai_factor, cropCoeff, refEvap):
    if irrigAmount !=0:
        surface_flux = (irrigAmount/daily_timesteps) + ((1-lai_factor)*cropCoeff*refEvap)
        surface_flux = surface_flux*daily_timesteps
        return surface_flux
    else:
        surface_flux = (1-lai_factor)*cropCoeff*refEvap
        return surface_flux

def Richards1D_vectorized(head, irrigAmount, cropCoeff, refEvap, soilPars, rooting_depth, lai_factor):
    ##The spatial parameters
    capCapacity = vg.capillaryCapacity(head, soilPars) 
    ##Creating an array for the flux  at each node
    flux = np.zeros(totalNodes+1)
    ##Top boundary condition
    #flux[totalNodes] = irrigAmount #Try adding the reference evapotranspiration
    flux[totalNodes] = determine_surface_flux(irrigAmount, lai_factor, cropCoeff, refEvap)

    ##Bottom boundary
    #For the free drainage boundary condition, the flux equals the hydraulic conductivity of the bottom node
    flux[0] = -1*vg.hydraulicConductivity(head[0], soilPars)

    ##Flux for the internal nodes
    #The hydraulic conductivity at the center of the compartments
    cntr_1 = np.arange(0, totalNodes-1)
    nodalHydCond = vg.hydraulicConductivity(head, soilPars)
    approxHydCond = (nodalHydCond[cntr_1 + 1] + nodalHydCond[cntr_1])/2.0
    cntr_2 = np.arange(1,totalNodes)
    flux[cntr_2] = -approxHydCond*(((head[cntr_1 + 1]-head[cntr_1])/nodal_distances[cntr_2 - 1]) + 1.0)
    sinkTerms = np.zeros((1, axialNodes))

    i= np.arange(0, 1)
    sinkTerms[i,:] = evaluate_sink_terms(refEvap, cropCoeff, rooting_depth, lai_factor)
    cntr_3 = np.arange(0, totalNodes)
    #The axial distance can be computed
    dhdt = ((-(flux[cntr_3 + 1] - flux[cntr_3])/ approx_distances[cntr_3]) - sinkTerms)/capCapacity
    return dhdt

def volMoistureAllNodes(head, soilPars):
    volMoistureInit = vg.volumetricMoisture(head, soilPars)
    volMoistureFin =volMoistureInit
    return volMoistureFin

def volMoistureAllNodes_1D(head, soilPars):
    volMoistureInit = vg.volumetricMoisture(head, soilPars)
    cMatrix = np.eye(totalNodes)
    volMoistureFin = np.dot(cMatrix, volMoistureInit)
    return volMoistureFin