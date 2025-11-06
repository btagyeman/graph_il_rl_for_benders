import numpy as np
import time
import sys
sys.path.append('../../trained_lstm_models')
from common.actual_field.richards_equation_field import volMoistureAllNodes as vwc_fun
from common.actual_field.temporal_spatial_parameters_field import *
from common.actual_field.crop_coefficient_calculation import generate_Kc
from lstm_mz1 import getRNNmodel_numpy_mz1
from common.actual_field.model_simulation import simulate_Richards_equation
from common.utils_season_long import evaluate_weather_uncertainy_ET,evaluate_weather_uncertainty_rain,compute_root_zone_moisture,obtain_soil_moisture
from common.estimator_design.ekf_algorithm import discreteTimeEKF
from common.estimator_design.richards_equation_ekf import  rz_moisture_alg,surface_moisture_alg
from helper_functions.gbd_mimpc import run_gbd_for_scheduler
totalDepth, axialNodes, totalNodes= spatialVariables_1D_act()
samplingTime, samplingTimeInternal, internalTimeSteps, totTimeSteps = temporalVariable()
soilPars_mz1 = pars_mz1()

#Load the weather conditions
_average_temps = np.loadtxt('./common/weather_data/dAVG_TEMP.txt')
_crop_coeff = generate_Kc(_average_temps)
_rain_act = np.loadtxt('./common/weather_data/dRAIN.txt')
_ref_Evap = np.loadtxt('./common/weather_data/dPET.txt')
_rain_predicted = _rain_act
_ref_Evap_predicted = _ref_Evap

for i in range(len(_ref_Evap_predicted)):
    if _ref_Evap_predicted[i]< 1.04:
        _ref_Evap_predicted[i] = 1.04
    if _ref_Evap_predicted[i] > 9.0:
        _ref_Evap_predicted[i] = 9.0
    pass
for i in range(len(_ref_Evap)):
    if _ref_Evap[i]< 1.04:
        _ref_Evap[i] = 1.04
    if _ref_Evap[i] > 9.0:
        _ref_Evap[i] = 9.0
    pass

#Load the scaling data for the simulation
data_min_mz1 = np.loadtxt('../../trained_lstm_models/model_weights/data_min_mz1_vrd_lai.txt')
data_max_mz1 = np.loadtxt('../../trained_lstm_models/model_weights/data_max_mz1_vrd_lai.txt')
referenceEvap_all_mz1 = _ref_Evap_predicted/(86400*1000)
referenceEvap_all_act_mz1 = _ref_Evap/(86400*1000)
cropCoefficient_all_mz1 = _crop_coeff
rain_all_predicted = -1*(_rain_predicted/1000)
rain_all_act = -1*(_rain_act/1000)
lai_factors_all = np.ones(180)

def run_irrigation_scheduler(self, k):
    print("\n\n")
    print("------------Open-loop simulation for day", k+1)
    x0_init_mz1 = np.array(self.statesList_mz1[k:k+self.sequenceLength])
    u_init_mz1  = np.array(self.irrigationList_mz1[k:k+self.sequenceLength-1])
    kc_init_mz1 = cropCoefficient_all_mz1[k:k+self.horizonLength_weather]
    et_init_mz1 = referenceEvap_all_mz1[k:k+self.horizonLength_weather]
    et_init_mz1_noisy = evaluate_weather_uncertainy_ET(et_init_mz1, len(et_init_mz1))
    # et_init_mz1_noisy = et_init_mz1 
    et_init_mz1_act = referenceEvap_all_act_mz1[k:k+self.horizonLength_weather]
    rain=rain_all_predicted[k:k+self.horizonLength_weather]
    rain_act=rain_all_act[k:k+self.horizonLength_weather]
    rooting_depth=self.rooting_depths[k:k+self.horizonLength_weather]
    rain_noisy=evaluate_weather_uncertainty_rain(rain, len(rain))
    # rain_noisy = rain 
    self.rain_noisy_all[k,:]=rain_noisy
    self.refEvap_noisy_all[k,:]=et_init_mz1_noisy
    lai_factors = lai_factors_all[k:k+self.horizonLength_weather]
    x0_scaled_mz1=(x0_init_mz1-data_min_mz1[0])/(data_max_mz1[0]-data_min_mz1[0])
    kc_scaled_mz1=(kc_init_mz1-data_min_mz1[2])/(data_max_mz1[2]-data_min_mz1[2])
    et_scaled_mz1=(et_init_mz1_noisy-data_min_mz1[3])/(data_max_mz1[3]-data_min_mz1[3])
    rd_scaled_mzs=(rooting_depth-data_min_mz1[4])/(data_max_mz1[4]-data_min_mz1[4])
    lai_scaled_mzs=(lai_factors-data_min_mz1[5])/(data_max_mz1[5]-data_min_mz1[5])

    # this step represents the main difference between the current code and the RL-based approach
    time_start = time.time()
    states_mz1,irrigAmnts_mz1,irrigDecs,ubd_final,lbd_final,optimal_values,graph_features,cut_order, eval_time_mp, mp_iter=run_gbd_for_scheduler(x0_scaled_mz1,u_init_mz1,self.guessesX,self.guessesU,self.guessesC,
    kc_scaled_mz1, et_scaled_mz1,rd_scaled_mzs,lai_scaled_mzs,rain,self.ubd,self.lbd,self.epsilon)
    time_end = time.time()
    self.total_eval_time.append(time_end - time_start)
    self.graph_features = graph_features
    self.cut_order = cut_order
    self.optimal_values = optimal_values
    self.eval_time_mp.extend(eval_time_mp)
    self.mp_count.append(mp_iter)
    self.ol_mp_time.append(sum(eval_time_mp))
    self.predictedIrrigations.append(irrigDecs)

    # update the guesses for the next iteration
    # self.guessesX = states_mz1 # unscaled 
    self.guessesX = (states_mz1 - data_min_mz1[0])/(data_max_mz1[0]-data_min_mz1[0])
    self.guessesX = self.guessesX.tolist()
    self.guessesU = np.array(irrigAmnts_mz1)
    self.guessesC = np.array(irrigDecs)

    #For monitoring purposes
    print(f"The initial irrigation amount for the first management zone is {irrigAmnts_mz1[0]}")
    # states_mz1_plot = np.zeros(len(states_mz1)+1)
    # states_mz1_plot[0] = x0_init_mz1[-1]
    # states_mz1_plot[1:] = states_mz1
    # fig, axs = plt.subplots(figsize=(8,8))
    # axs.plot(states_mz1_plot)
    # axs.plot(self.UpperZone_mz1*np.ones(len(states_mz1_plot)))
    # axs.plot(self.LowerZone_mz1*np.ones(len(states_mz1_plot)))
    # plt.show()

    self.irrigationList_mz1+=[irrigAmnts_mz1[0]]
    self.prescribedIrrigation_closedLoop_mz1+=[irrigAmnts_mz1[0]]
    self.stateTrajectory_openLoop_mz1+=[states_mz1[0]]
    self.irrigationDecisions_closedLoop+=[irrigDecs[0]]
    
    #Actual plant simulation (RE) - MZ 1
    x_currentSE_mz1=self.allStates_mz1[k,:]
    x_currentSE_mz1_est=self.allStates_mz1_est[k,:]
    u_currentSE_mz1=(irrigAmnts_mz1[0]+rain_act[4])/86400
    kc_currentSE_mz1=kc_init_mz1[4]
    et_currentSE_mz1=et_init_mz1_act[4]
    rd_currentSE_mz1=rooting_depth[4]
    lai_currentSE_mz1=lai_factors[4]
    currentStatesSE_mz1=simulate_Richards_equation(x_currentSE_mz1,u_currentSE_mz1,et_currentSE_mz1,kc_currentSE_mz1,4,rd_currentSE_mz1,lai_currentSE_mz1,soilPars_mz1)
    currentStatesSE_mz1_est=simulate_Richards_equation(x_currentSE_mz1_est,u_currentSE_mz1,et_currentSE_mz1, kc_currentSE_mz1,4, rd_currentSE_mz1,lai_currentSE_mz1,soilPars_mz1)
    currentStatesSE_mz1_vol=obtain_soil_moisture(currentStatesSE_mz1, soilPars_mz1,vwc_fun)
    rootzone_moist_mz1=compute_root_zone_moisture(currentStatesSE_mz1_vol, rd_currentSE_mz1)
    self.allStates_mz1[k+1,:]=currentStatesSE_mz1

    #LSTM Model Simulation - MZ1    
    x_current_mz1 = np.array(self.statesList_mz1[k:k+self.sequenceLength])
    x_current_scaled_mz1 = (x_current_mz1 - data_min_mz1[0])/(data_max_mz1[0]-data_min_mz1[0])
    u_current_mz1 = np.array(self.irrigationList_mz1[k:k+self.sequenceLength]) + rain_noisy[0:0+self.sequenceLength] #Rain with noisy should be used for the model simulation
    u_current_mz1 = u_current_mz1/86400
    u_current_scaled_mz1 = (u_current_mz1 - data_min_mz1[1])/(data_max_mz1[1] - data_min_mz1[1])
    #Ensure that the scaled irrigation rate ranges from 0 to 1
    u_current_scaled_mz1[u_current_scaled_mz1<0]=0
    u_current_scaled_mz1[u_current_scaled_mz1>1]=1

    kc_current_mz1 = kc_scaled_mz1[0:0+self.sequenceLength]
    et_current_mz1 = et_scaled_mz1[0:0+self.sequenceLength]
    rd_current_mz1 = rd_scaled_mzs[0:0+self.sequenceLength]
    lai_current_mz1 = lai_scaled_mzs[0:0+self.sequenceLength]
    currentState_mz1 = getRNNmodel_numpy_mz1(x_current_scaled_mz1,u_current_scaled_mz1,kc_current_mz1,et_current_mz1,rd_current_mz1,lai_current_mz1)
    currentState_unscaled_mz1 = (currentState_mz1*(data_max_mz1[0]-data_min_mz1[0]))+data_min_mz1[0]

    measurement_mz1=self.factor_mz1*surface_moisture_alg(currentStatesSE_mz1[10:], soilPars_mz1)
    x_currentSE_mz1_updated,covariance_mz1=discreteTimeEKF(x_currentSE_mz1_est[10:],currentStatesSE_mz1_est[10:],u_currentSE_mz1,kc_currentSE_mz1,
        et_currentSE_mz1,1.0,measurement_mz1,self.covariance_mz1,self.process_covariance,self.measurement_covariance,soilPars_mz1)
    
    self.covariance_mz1 = covariance_mz1  # Update the covariance for the next iteration
    currentStatesSE_mz1_est=x_currentSE_mz1_updated
    currentStatesSE_mz1_est=np.concatenate((currentStatesSE_mz1[0:10],currentStatesSE_mz1_est))
    self.statesList_mz1+=[rz_moisture_alg(currentStatesSE_mz1_est[10:],soilPars_mz1)]
    self.allStates_mz1_est[k+1,:] = currentStatesSE_mz1_est
    self.stateTrajectory_closedLoop_mz1+=[rootzone_moist_mz1]
    pass