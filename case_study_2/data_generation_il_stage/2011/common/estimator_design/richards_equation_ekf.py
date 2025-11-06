# This module contains the casidi version of the finite difference model for the 1D Richards Equation (RE). 
# The 1D RE is expressed as C(h)dh/dt = d/dz(K(h)d/dz(h+z) where the terms have the following meanings
# C(h) -  differential water capacity/ capillary capacity
# h is -  the soil water pressure head
# K(h) -  is the unsaturated hydraulic conductivity
# z is -  the vertival coordinate directed upward

# Assumptions
# 1. The field is homogeneous (i.e same soil type) and no slope exists.
# 2. This module does not include the crop model.

# Boundary Conditions
# 1. Top BC    - Flux boundary condition [dh/dz= -1 - irrigation_amount/K(h)]
# 2. Bottom BC - Free Drainage [d(h+z)/dz= 1]

# The dependence of C, and K on h can be found in the van_Genuchten_relations module.

import numpy as np 
from casadi import *
from common.estimator_design.temporal_spatial_ekf import spatialVariables_1D, temporalVariable,generatingCoordinates_1D, CMatrixAllMeasurements
import common.estimator_design.van_Genuchten_ekf as vg

totalDepth, axialNodes, axialDistance, totalNodes = spatialVariables_1D()  #Spatial parameters 

nodal_distances = np.zeros(axialNodes-1)

daily_timesteps = 1 ## here, the daily times steps should be equal to the factor
factor = 1
for i in range(0, axialNodes-1):
    nodal_distances[i] = 0.025
    pass

approx_distances = np.zeros(axialNodes)
for l in range(0, axialNodes):
    approx_distances[l] = 0.025
    pass

actual_distances_50cm = np.zeros(axialNodes)
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

def rz_moisture(axialStates):
    rootZone_vol_moist = 0.10*((1/6)*(axialStates[0] +axialStates[1] +  axialStates[2] +  axialStates[3] +  axialStates[4] + axialStates[5])) + \
    0.20*((1/6)*(axialStates[5] +   axialStates[6] +  axialStates[7] +  axialStates[8] +  axialStates[9] + axialStates[10])) + \
    0.30*((1/6)*(axialStates[10] +  axialStates[11] + axialStates[12] + axialStates[13] + axialStates[14] +axialStates[15])) + \
    0.40*((1/6)*(axialStates[15] +  axialStates[16] + axialStates[17] + axialStates[18] + axialStates[19] +axialStates[20]))
    return rootZone_vol_moist

def evaluate_sink_terms(refEvap, cropCoeff, lai_factor, axial_node):
    rooting_depth = 0.50 
    sinkTerm = (2*(lai_factor*refEvap*cropCoeff)/rooting_depth)*(1 -((rooting_depth - actual_distances_50cm[axial_node])/rooting_depth))    
    return sinkTerm

def determine_surface_flux(irrigAmount, lai_factor, cropCoeff, refEvap):
    surface_flux = (irrigAmount*factor) + ((1-lai_factor)*cropCoeff*refEvap/factor)
    surface_flux = surface_flux/factor
    return surface_flux

def RichardsPolar_1D(head, irrigAmount, cropCoeff, refEvap,lai_factor, soilPars):
    coordinates  = generatingCoordinates_1D(axialNodes)
    dhdt = SX.zeros(totalNodes)
    for i in range(0,totalNodes):
        currentState = head[i]
        currentNode  = coordinates[i]
        curAxialNode = currentNode[-1] #The last coordinate represents the axial node

        if curAxialNode == 0: #This represents the bottom of the field
            stateSouth = currentState
            stateNorth = head[i + 1]
            hydCondNorth = vg.hydraulicConductivityApprox(currentState, stateNorth,soilPars)
            hydCondSouth = vg.hydraulicConductivityApprox(currentState, stateSouth,soilPars)
            fluxNorth = hydCondNorth*(((stateNorth   - currentState)/axialDistance) + 1)
            fluxSouth = hydCondSouth*(((currentState - stateSouth)/axialDistance) + 1)
            axialComponent = (1/axialDistance) * (fluxNorth - fluxSouth)

        elif curAxialNode == axialNodes - 1: # This represents the surface of the field where the irrigation is applied
            stateNorth = 0
            stateSouth = head[i - 1]
            hydCondNorth = 0
            hydCondSouth =  vg.hydraulicConductivityApprox(currentState, stateSouth,soilPars)
            fluxNorth = -determine_surface_flux(irrigAmount, lai_factor, cropCoeff, refEvap)
            fluxSouth = hydCondSouth*(((currentState-stateSouth)/axialDistance) + 1)
            axialComponent = (1/axialDistance)*(fluxNorth - fluxSouth)

        else: #Interior nodes
            stateNorth = head[i + 1]
            stateSouth = head[i - 1]
            hydCondNorth = vg.hydraulicConductivityApprox(currentState, stateNorth,soilPars)
            hydCondSouth = vg.hydraulicConductivityApprox(currentState, stateSouth,soilPars)
            fluxNorth = hydCondNorth*(((stateNorth   - currentState)/axialDistance) + 1)
            fluxSouth = hydCondSouth*(((currentState - stateSouth)/axialDistance) + 1)
            axialComponent = (1/axialDistance)*(fluxNorth - fluxSouth)
        sinkTerm = evaluate_sink_terms(refEvap, cropCoeff, lai_factor, curAxialNode)
        spatialApprox = (1/vg.capillaryCapacity(currentState, soilPars))*(axialComponent - sinkTerm)
        dhdt[i] = spatialApprox
    DhDt = dhdt
    return DhDt

def volMoistureAllNodes_1D(head, azimuthalNode, radialNode):
    soilPars = obtain_soil_parameters(azimuthalNode, radialNode)
    volMoistureInit = vg.volumetricMoisture(head, soilPars)
    cMatrix = CMatrixAllMeasurements(totalNodes)
    volMoistureFin = mtimes(cMatrix, volMoistureInit)
    return volMoistureFin

##For this model,the cMatrix is defined as follows
def cMatrix_1D(axialNodes):
    cMatrix = np.zeros(axialNodes)
    for i in range(axialNodes):
        cMatrix[i] = (1/axialNodes)
    cMatrix = np.reshape(cMatrix, (1, totalNodes))
    return cMatrix

def volMoistureSelected_1D(cMatrix, head, radialNode, azimuthalNode):
    soilPars = obtain_soil_parameters(azimuthalNode, radialNode)
    volMoistureInit = vg.volumetricMoisture(head, soilPars)
    volMoistureFin = mtimes(cMatrix, volMoistureInit)
    return volMoistureFin

def rz_moisture_sym(axialStates_sym, soilPars):
    axialStates_sym = vg.volumetricMoisture(axialStates_sym, soilPars)
    return rz_moisture(axialStates_sym)

def surface_moisture_sym(axialStates_sym, soilPars):
    axialStates_sym = vg.volumetricMoisture(axialStates_sym, soilPars) # first compute the spatial vwc
    selection_matrix = DM.zeros(totalNodes)
    for i in range(10,totalNodes):
        selection_matrix[i] = 1/11
        pass 
    surface_moisture = mtimes(selection_matrix.T,axialStates_sym)
    return surface_moisture


#This should be the output equation for the state estimation
def rz_moisture_alg(axialStates, soilPars):
    #axial states go in initially in pressure head form
    axialStates = vg.volumetricMoisture(axialStates, soilPars) 
    axialStates = axialStates.full().ravel()
    return rz_moisture(axialStates)


#This should be the output equation for the state estimation
def surface_moisture_alg(axialStates, soilPars):
    #axial states go in initially in pressure head form
    axialStates = vg.volumetricMoisture(axialStates, soilPars) 
    axialStates = axialStates.full().ravel()
    selection_matrix = np.zeros(totalNodes)
    for i in range(10,totalNodes):
        selection_matrix[i] = 1/11
        pass
    surface_moisture = selection_matrix.T @ axialStates 
    return surface_moisture