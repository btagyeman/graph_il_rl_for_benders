import numpy as np
from scipy import integrate
from common.actual_field.richards_equation_field import *
from common.actual_field.temporal_spatial_parameters_field import *
from common.actual_field.van_Genuchten_relations import pressureHead, volumetricMoisture
totalDepth, axialNodes, totalNodes= spatialVariables_1D_act()

def ODE(head, timeArray, irrigAmount, cropCoeff, refEvap, soilPars, rooting_depth, lai_factor):
         return Richards1D_vectorized(head, irrigAmount, cropCoeff, refEvap, soilPars, rooting_depth, lai_factor)
     
def simulate_Richards_equation_all(initial_condition, irrigation_rates_daily, reference_evap_daily, crop_coeff_daily, time_steps_daily, rooting_depth, soilPars):
    samplingTime, samplingTimeInternal, internalTimeSteps, totalTimeSteps_daily, totalTimeSteps = temporal_discretization(time_steps_daily)
    head_array = np.zeros((totalTimeSteps+1, totalNodes))
    volMoisture = np.zeros((totalTimeSteps+1, totalNodes))
    irrigation_rates = np.zeros(totalTimeSteps)
    evapotranspiration =np.zeros(totalTimeSteps)
    crop_coeff = np.zeros(totalTimeSteps)
    head_array[0] = initial_condition
    volMoisture[0] = volMoistureAllNodes_1D(initial_condition, soilPars)
    
    for i in range(len(irrigation_rates_daily)):
        irrigation_rates[int(i*time_steps_daily)] = irrigation_rates_daily[i]*time_steps_daily
        pass
    
    for i in range(len(reference_evap_daily)):
        crop_coeff[int(i*totalTimeSteps_daily):int((i+1)*totalTimeSteps_daily)] = crop_coeff_daily[i]*np.ones(totalTimeSteps_daily)
        evapotranspiration[int(i*totalTimeSteps_daily):int((i+1)*totalTimeSteps_daily)] = (reference_evap_daily[i]*1)*np.ones(totalTimeSteps_daily)
        pass
    
    for i in range(1,totalTimeSteps+1):
        #print(i)    
        sol = integrate.solve_ivp(fun=lambda t, y: ODE(y, t, irrigation_rates[i-1],crop_coeff[i-1], evapotranspiration[i-1], soilPars),t_span=[0,samplingTimeInternal],\
                                  y0=tuple(head_array[i-1]),method='LSODA')
        head_array[i,:]=sol.y[:,-1]
        volMoisture[i,:] = volMoistureAllNodes_1D(head_array[i,:], soilPars).ravel()       
        pass
    return head_array, volMoisture

def simulate_Richards_equation(initial_condition, irrigation_rates_daily, reference_evap_daily, crop_coeff_daily, time_steps_daily, rooting_depth, lai_factor, soilPars):
    samplingTime, samplingTimeInternal, internalTimeSteps, totalTimeSteps_daily, totalTimeSteps = temporal_discretization(time_steps_daily)
    head_array = np.zeros((totalTimeSteps+1, totalNodes))
    irrigation_rates = np.zeros(totalTimeSteps)
    evapotranspiration =np.zeros(totalTimeSteps)
    crop_coeff = np.zeros(totalTimeSteps)
    head_array[0] = initial_condition
    
    for i in range(1):
        irrigation_rates[int(i*time_steps_daily)] = irrigation_rates_daily*time_steps_daily
        pass
    
    for i in range(1):
        crop_coeff[int(i*totalTimeSteps_daily):int((i+1)*totalTimeSteps_daily)] = crop_coeff_daily*np.ones(totalTimeSteps_daily)
        evapotranspiration[int(i*totalTimeSteps_daily):int((i+1)*totalTimeSteps_daily)] = (reference_evap_daily*1)*np.ones(totalTimeSteps_daily)
        pass
        
    for i in range(1,totalTimeSteps+1):   
        sol = integrate.solve_ivp(fun=lambda t, y: ODE(y, t, irrigation_rates[i-1], crop_coeff[i-1], evapotranspiration[i-1], soilPars, rooting_depth, lai_factor),
        t_span=[0,samplingTimeInternal], y0=tuple(head_array[i-1]),method='LSODA')
        head_array[i,:]=sol.y[:,-1]       
        pass
    return head_array[-1,:]

def SimulateModel(state, action, refEvap, cropCoeff, rooting_depth, lai_factor, soilPars):
    #Perform the simulation and return the last values
    state_head = pressureHead(state, soilPars) # Convert volumetric moisture content to pressure head
    irrig_rate = (action/86400)
    nextState_head = simulate_Richards_equation(state_head, irrig_rate, refEvap/(86400*1000), cropCoeff, 4, rooting_depth, lai_factor, soilPars)
    nextState_vol_moist = volumetricMoisture(nextState_head, soilPars) #Convert pressure head to volumetric soil moisture
    return nextState_vol_moist