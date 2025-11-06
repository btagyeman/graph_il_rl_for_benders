
#This module contains the functions that simulate the environment for the upper agent and the lower agents (3 lower agents, one for each management zone)
from richards_equation_scheduler import Richards1D_vectorized, volMoistureAllNodes_1D
import numpy as np
from temporal_spatial_scheduler import spatialVariables_1D, pars_mz1, pars_mz2, pars_mz3, temporal_discretization
from scipy import integrate
from van_Genuchten_scheduler import pressureHead, volumetricMoisture

# Load the parameters of the various management zones
soilPars_mz1 = pars_mz1()
soilPars_mz2 = pars_mz2()
soilPars_mz3 = pars_mz3()
totalDepth, axialNodes, totalNodes = spatialVariables_1D()  # Spatial parameters


def ODE(head, timeArray, irrigAmount, refEvap, cropCoeff, lai_factor, soilPars):
    return Richards1D_vectorized(head, irrigAmount, refEvap, cropCoeff, lai_factor, soilPars)

def simulate_Richards_equation(initial_condition, irrigation_rates_daily, reference_evap_daily, crop_coeff_daily, time_steps_daily, lai_factor, soilPars):
    samplingTime, samplingTimeInternal, internalTimeSteps, totalTimeSteps_daily, totalTimeSteps = temporal_discretization(time_steps_daily)
    head_array = np.zeros((totalTimeSteps+1, totalNodes))
    volMoisture = np.zeros((totalTimeSteps+1, totalNodes))
    irrigation_rates = np.zeros(totalTimeSteps)
    evapotranspiration = np.zeros(totalTimeSteps)
    crop_coeff = np.zeros(totalTimeSteps)
    head_array[0] = initial_condition
    # volMoisture[0] = volMoistureAllNodes_1D(initial_condition, soilPars).full().ravel()
    volMoisture[0] = volMoistureAllNodes_1D(initial_condition, soilPars).ravel()

    for i in range(1):
        irrigation_rates[int(i*time_steps_daily)] = irrigation_rates_daily*time_steps_daily
        pass

    for i in range(1):
        crop_coeff[int(i*totalTimeSteps_daily):int((i+1)*totalTimeSteps_daily)] = crop_coeff_daily*np.ones(totalTimeSteps_daily)
        evapotranspiration[int(i*totalTimeSteps_daily):int((i+1)*totalTimeSteps_daily)] = (reference_evap_daily*1)*np.ones(totalTimeSteps_daily)
        pass

    for i in range(1, totalTimeSteps+1):
        sol = integrate.solve_ivp(fun=lambda t, y: ODE(y, t, irrigation_rates[i-1], crop_coeff[i-1], evapotranspiration[i-1], lai_factor, soilPars),
                                t_span=[0, samplingTimeInternal], y0=tuple(head_array[i-1]), method='LSODA')
        head_array[i, :] = sol.y[:, -1]
        volMoisture[i, :] = volMoistureAllNodes_1D(head_array[i, :], soilPars).ravel()
        pass
    return head_array[-1, :]

# Environment for the upper agent
def SimulateModel_UA(state, actions, refEvap, cropCoeff, lai,rain):
    states_mz1 = state[0*axialNodes:1*axialNodes]
    states_mz2 = state[1*axialNodes:2*axialNodes]
    states_mz3 = state[2*axialNodes:3*axialNodes]
    action_mz1 = actions[0]
    action_mz2 = actions[1]
    action_mz3 = actions[2]
    state_head_mz1 = pressureHead(states_mz1, soilPars_mz1)
    irrig_rate_mz1 = (action_mz1/86400) + (rain/86400)
    state_head_mz2 = pressureHead(states_mz2, soilPars_mz2)
    irrig_rate_mz2 = (action_mz2/86400) + (rain/86400)
    state_head_mz3 = pressureHead(states_mz3, soilPars_mz3)
    irrig_rate_mz3 = (action_mz3/86400) + (rain/86400)

    nextState_head_mz1 = simulate_Richards_equation(state_head_mz1, irrig_rate_mz1, refEvap/(86400*1000), cropCoeff, 4, lai, soilPars_mz1)
    nextState_head_mz2 = simulate_Richards_equation(state_head_mz2, irrig_rate_mz2, refEvap/(86400*1000), cropCoeff, 4, lai, soilPars_mz2)
    nextState_head_mz3 = simulate_Richards_equation(state_head_mz3, irrig_rate_mz3, refEvap/(86400*1000), cropCoeff, 4, lai, soilPars_mz3)

    nextState_vol_moist_mz1 = volumetricMoisture(nextState_head_mz1, soilPars_mz1)
    nextState_vol_moist_mz2 = volumetricMoisture(nextState_head_mz2, soilPars_mz2)
    nextState_vol_moist_mz3 = volumetricMoisture(nextState_head_mz3, soilPars_mz3)

    agent_state_all = np.zeros(int(3*axialNodes))
    agent_state_all[0*axialNodes:1*axialNodes] = nextState_vol_moist_mz1
    agent_state_all[1*axialNodes:2*axialNodes] = nextState_vol_moist_mz2
    agent_state_all[2*axialNodes:3*axialNodes] = nextState_vol_moist_mz3
    return agent_state_all

# Environment for the agent that selects the irrigation rate for management zone 1
def SimulateModel_LA_a(state, action, refEvap, cropCoeff, lai, rain):
    state_head = pressureHead(state, soilPars_mz1)
    irrig_rate = (action/86400) + (rain/86400)
    nextState_head = simulate_Richards_equation(state_head, irrig_rate, refEvap/(86400*1000), cropCoeff, 4, lai, soilPars_mz1)
    nextState_vol_moist = volumetricMoisture(nextState_head, soilPars_mz1)
    return nextState_vol_moist

# Environment for the agent that selects the irrigation rate for management zone 2
def SimulateModel_LA_b(state, action, refEvap, cropCoeff, lai, rain):
    state_head = pressureHead(state, soilPars_mz2)
    irrig_rate = (action/86400) + (rain/86400)
    nextState_head = simulate_Richards_equation(state_head, irrig_rate, refEvap/(86400*1000), cropCoeff, 4, lai, soilPars_mz2)
    nextState_vol_moist = volumetricMoisture(nextState_head, soilPars_mz2)
    return nextState_vol_moist

# Environment for the agent that selects the irrigation rate for management zone 3
def SimulateModel_LA_c(state, action, refEvap, cropCoeff, lai, rain):
    state_head = pressureHead(state, soilPars_mz3)
    irrig_rate = (action/86400) + (rain/86400)
    nextState_head = simulate_Richards_equation(state_head, irrig_rate, refEvap/(86400*1000), cropCoeff, 4, lai, soilPars_mz3)
    nextState_vol_moist = volumetricMoisture(nextState_head, soilPars_mz3)
    return nextState_vol_moist
