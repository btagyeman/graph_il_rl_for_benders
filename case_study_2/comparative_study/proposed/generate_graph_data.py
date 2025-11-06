import numpy as np
from pyomo.environ import *
import os 
from matplotlib import pyplot as plt
from common.actual_field.temporal_spatial_parameters_field import *
from common.actual_field.richards_equation_field import volMoistureAllNodes as vwc_fun
from common.utils_season_long import compute_root_zone_moisture,obtain_soil_moisture
from helper_functions.season_long_simulation import run_irrigation_scheduler
totalDepth, axialNodes, totalNodes= spatialVariables_1D_act()
os.makedirs('./results', exist_ok=True)
soilPars_mz1 = pars_mz1()

#Load the scaling data for the simulation
data_min_mz1 = np.loadtxt('../../trained_lstm_models/model_weights/data_min_mz1_vrd_lai.txt')
data_max_mz1 = np.loadtxt('../../trained_lstm_models/model_weights/data_max_mz1_vrd_lai.txt')
_rain_act = np.loadtxt('./common/weather_data/dRAIN.txt')

#Load the initial states and uncontrolled inputs (i.e. ET and Crop Coeff)
initialState_mz1 = np.loadtxt("./common/initial_states/Init_state_rz_mz1_vrd_lai.txt")
initialIrrigation_mz1 = np.loadtxt("./common/initial_states/Init_irrig_mz1_vrd_lai.txt")
initialIrrigation_mz1 = initialIrrigation_mz1*86400
allHeadValues_mz1 = np.loadtxt("./common/initial_states/Init_state_all_mz1_vrd_lai.txt")

class SeasonLongInvestigation():
    def __init__(self):
        self.measurement_covariance = 19.25 
        self.process_covariance = 0.05*np.eye(21)
        self.covariance_mz1 = 15.9*np.ones((21,21))
        self.simulation_period = 123
        self.time_instants = np.arange(self.simulation_period) 
        self.prediction_horizon = 14 
        self.sequenceLength = 5 
        self.horizonLength_weather = int(self.sequenceLength + self.prediction_horizon)
        self.rooting_depths = 0.5*np.ones(160)
        self.rain_noisy_all=np.zeros((self.simulation_period,self.horizonLength_weather))
        self.refEvap_noisy_all=np.zeros((self.simulation_period,self.horizonLength_weather))
        self.guessesX=np.array([0.224,0.252,0.246,0.242,0.234,0.232,0.229,0.226,0.224,0.221,0.219,0.216,0.213,0.210])
        self.guessesU=-0.003887*np.ones(self.prediction_horizon)
        self.guessesC=np.ones(self.prediction_horizon) 
        self.statesList_mz1 = []
        self.irrigationList_mz1 = []
        self.set_initial_vwc_and_irrig()
        self.prescribedIrrigation_closedLoop_mz1 = []
        self.stateTrajectory_closedLoop_mz1 = []
        self.stateTrajectory_openLoop_mz1 = []
        self.irrigationDecisions_closedLoop = []
        self.predictedIrrigations = []
        self.stateTrajectory_closedLoop_mz1.append(initialState_mz1[-1])
        self.stateTrajectory_openLoop_mz1.append(compute_root_zone_moisture(obtain_soil_moisture(1.05*allHeadValues_mz1,soilPars_mz1,vwc_fun),0.50))
        self.factor_mz1=compute_root_zone_moisture(obtain_soil_moisture(1.05*allHeadValues_mz1,soilPars_mz1,vwc_fun),0.50)/initialState_mz1[4]
        self.allStates_mz1=np.zeros((self.simulation_period+1, totalNodes))
        self.allStates_mz1_est=np.zeros((self.simulation_period+1, totalNodes))
        self.allStates_mz1[0,:]=allHeadValues_mz1
        self.allStates_mz1_est[0,:]=1.05*allHeadValues_mz1
        self.eval_time_mp = []
        self.mp_count = []
        self.ol_mp_time = []
        self.total_eval_time = []
        self.prediced_binary_variables = [] 
        self.graph_features=None 
        self.cut_order=None  
        self.optimal_values=None
        self.n_variables = 14 
        self.lbd = -10e18
        self.ubd = 10e18
        self.LowerZone_mz1 = 0.200 
        self.UpperZone_mz1 = 0.280 
        self.epsilon = 0.5
        # self.epsilon = 1.2
        self.R_c = 5 
        self.R_u = -50
        pass 

    def set_initial_vwc_and_irrig(self):
        for j in range(self.sequenceLength-1):
            self.statesList_mz1+=[initialState_mz1[j]]
            self.irrigationList_mz1+=[initialIrrigation_mz1[j]]
            pass 
        self.statesList_mz1+=[compute_root_zone_moisture(obtain_soil_moisture(1.05*allHeadValues_mz1,soilPars_mz1,vwc_fun),0.50)]
        return 
    
    def plot_trajectory(self):
        u_prescribed_mz1 = np.array(self.prescribedIrrigation_closedLoop_mz1)*1000
        timeStamps = np.arange(1, self.simulation_period+2)
        timeStamps_input = np.arange(1, self.simulation_period+1)
        fig, axs = plt.subplots(2, figsize=(10,10))
        axs[0].bar(timeStamps_input,-u_prescribed_mz1, width = 1.2, facecolor=None)
        axs[0].bar(timeStamps_input,_rain_act[4:len(u_prescribed_mz1)+4], color='red', width = 1.0)
        axs[0].legend(['Proposed', 'Rain'])
        axs[0].set_title('Irrigation -  MZ 1 ')
        axs[0].set_ylabel('Irrigation (mm/day)')
        axs[0].set_xlabel('Time (days)')
        axs[0].set_xlim(xmin=1)
        axs[1].plot(timeStamps,self.stateTrajectory_closedLoop_mz1,color='blue', linewidth=1.8)
        axs[1].plot(timeStamps,self.UpperZone_mz1*np.ones(len(self.stateTrajectory_openLoop_mz1)), color='darkcyan', linestyle = 'dashdot', linewidth=2.5)
        axs[1].plot(timeStamps,self.LowerZone_mz1*np.ones(len(self.stateTrajectory_openLoop_mz1)), color='darkmagenta', linestyle = 'dashdot', linewidth=2.5)
        axs[1].plot(timeStamps,0.12*np.ones(len(self.stateTrajectory_openLoop_mz1)), color='red', linestyle='dashdot', linewidth=2.5)
        axs[1].legend(['state Trajectory', 'FC', 'Threshold', 'PWP'])
        axs[1].set_title('Root Zone Soil Moisture - MZ 1')
        axs[1].set_ylabel('Volumetric moisture')
        axs[1].set_xlabel('Time (days)')
        axs[1].set_xlim(xmin=1)
        fig.tight_layout()
        fig.savefig('./results/Irrigation_trajectory_mz1.png', dpi=300)
        plt.show()
        return

    def evaluate_gbd(self):
        for k in self.time_instants:
            run_irrigation_scheduler(self, k)
            pass 
        return 
    
    def save_data(self):
        np.savetxt("./results/eval_time_mp.txt", np.array(self.eval_time_mp))
        np.savetxt("./results/total_eval_time.txt", np.array(self.total_eval_time))
        np.savetxt("./results/irrigation_decisions_closed_loop.txt", np.array(self.irrigationDecisions_closedLoop))
        np.savetxt("./results/irrigation_list_mz1.txt", np.array(self.irrigationList_mz1))
        np.savetxt("./results/state_trajectory_closed_loop_mz1.txt", np.array(self.stateTrajectory_closedLoop_mz1))
        np.savetxt("./results/mp_count.txt", np.array(self.mp_count))
        np.savetxt("./results/ol_mp_time.txt", np.array(self.ol_mp_time))
        np.savetxt("./results/predicted_variables_agent.txt", np.array(self.prediced_binary_variables))
        return 
    
eval_instance=SeasonLongInvestigation()
eval_instance.evaluate_gbd()
eval_instance.save_data()
eval_instance.plot_trajectory()
