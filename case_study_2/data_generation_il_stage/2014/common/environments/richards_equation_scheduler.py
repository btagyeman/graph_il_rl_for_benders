# This module contains the vectorized finite difference model for the 1D Richards Equation (RE).
# The 1D RE is expressed as C(h)dh/dt = d/dz(K(h)d/dz(h+z) where the terms have the following meanings
# C(h) -  differential water capacity/ capillary capacity
# h is -  the soil water pressure head
# K(h) -  is the unsaturated hydraulic conductivity
# z is -  the vertical coordinate directed upward

# Boundary Conditions6
# 1. Top BC    - Flux boundary condition [dh/dz= -1 - (irrigation_amount-EV)/K(h)] where EV is the evaporation rate
# 2. Bottom BC - Free Drainage [d(h+z)/dz= 1]

# The dependence of C, and K on h can be found in the van_Genucthen module.
#----by Bernard Twum Agyeman (agyeman@ualberta.ca)
#Imports

import  numpy as np 
from temporal_spatial_scheduler import spatialVariables_1D
import van_Genuchten_scheduler as vg
totalDepth, axialNodes, totalNodes = spatialVariables_1D() 
nodal_distances = np.zeros(totalNodes-1)

daily_timesteps = 4
for i in range(0, totalNodes-1):
    nodal_distances[i] = 0.025
    pass

approx_distances = np.zeros(totalNodes)
for l in range(0, totalNodes):
    approx_distances[l] = 0.025
    pass

actual_distances_50cm = np.zeros(totalNodes)
actual_distances_50cm[0]  =  0.000
actual_distances_50cm[1]  =  0.025
actual_distances_50cm[2]  =  0.050
actual_distances_50cm[3]  =  0.075
actual_distances_50cm[4]  =  0.100
actual_distances_50cm[5]  =  0.125
actual_distances_50cm[6]  =  0.150
actual_distances_50cm[7]  =  0.175
actual_distances_50cm[8]  =  0.200
actual_distances_50cm[9]  =  0.225
actual_distances_50cm[10] =  0.250
actual_distances_50cm[11] =  0.275
actual_distances_50cm[12] =  0.300
actual_distances_50cm[13] =  0.325
actual_distances_50cm[14] =  0.350
actual_distances_50cm[15] =  0.375
actual_distances_50cm[16] =  0.400
actual_distances_50cm[17] =  0.425
actual_distances_50cm[18] =  0.450
actual_distances_50cm[19] =  0.475 
actual_distances_50cm[20] =  0.500

def evaluate_sink_terms(refEvap, cropCoeff, lai_factor):
    rooting_depth = 0.50 
    sinkTerms = np.zeros(totalNodes)
    sinkTerms[0:] = (2*(lai_factor*refEvap*cropCoeff)/rooting_depth)*(1 -((rooting_depth - actual_distances_50cm[0:])/rooting_depth))    
    return sinkTerms

def determine_surface_flux(irrigAmount, lai_factor, cropCoeff, refEvap):
    if irrigAmount !=0:
        surface_flux = (irrigAmount/daily_timesteps) + ((1-lai_factor)*cropCoeff*refEvap)
        surface_flux = surface_flux*daily_timesteps
        return surface_flux
    else:
        surface_flux = (1-lai_factor)*cropCoeff*refEvap
        return surface_flux

def Richards1D_vectorized(head, irrigAmount, cropCoeff, refEvap, lai_factor, soilPars):
    ##The spatial parameters
    capCapacity = vg.capillaryCapacity(head, soilPars) 
    ##Creating an array for the flux  at each node
    flux = np.zeros(totalNodes+1)
    ##Top boundary condition
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
    sinkTerms[i,:] = evaluate_sink_terms(refEvap, cropCoeff,lai_factor)
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
