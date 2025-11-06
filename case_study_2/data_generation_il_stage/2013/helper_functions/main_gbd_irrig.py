
import numpy as np
from common.actual_field.richards_equation_field import volMoistureAllNodes
from common.actual_field.temporal_spatial_parameters_field import *
from common.actual_field.crop_coefficient_calculation import generate_Kc
from gbd_mimpc import run_gbd_for_scheduler

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

def compute_root_zone_moisture(Xk, rooting_depth):
    if rooting_depth == 1.00: 
        root_zone_head = 0.10*((1/6)*(Xk[0] + Xk[1] + Xk[2] + Xk[3] + Xk[4] + Xk[5]))  + \
            0.20*((1/6)*(Xk[5] + Xk[6] + Xk[7] + Xk[8] + Xk[9] + Xk[10])) + \
            0.30*((1/11)*(Xk[10] + Xk[11] + Xk[12] + Xk[13] + Xk[14] + Xk[15] + Xk[16] + Xk[17] + Xk[18] + Xk[19] + Xk[20])) + \
            0.40*((1/11)*(Xk[20] + Xk[21] + Xk[22] + Xk[23] + Xk[24] + Xk[25] + Xk[26] + Xk[27] + Xk[28] + Xk[29] + Xk[30]))
    else:    
        root_zone_head = 0.10*((1/6)*(Xk[10] + Xk[11] + Xk[12] + Xk[13] + Xk[14] + Xk[15])) +\
            0.20*((1/6)*(Xk[15] + Xk[16] + Xk[17] + Xk[18] + Xk[19] + Xk[20])) +\
            0.30*((1/6)*(Xk[20] + Xk[21] + Xk[22] + Xk[23] + Xk[24] + Xk[25])) + \
            0.40*((1/6)*(Xk[25] + Xk[26] + Xk[27] + Xk[28] + Xk[29] + Xk[30]))
    return root_zone_head   

#Converts the root zone pressure head to soil moisture
def obtain_soil_moisture(head, soilPars):
    vol_moist = volMoistureAllNodes(head, soilPars)
    return vol_moist

#Load the scaling data for the simulation
data_min_mz1 = np.loadtxt('../../trained_lstm_models/model_weights/data_min_mz1_vrd_lai.txt')
data_max_mz1 = np.loadtxt('../../trained_lstm_models/model_weights/data_max_mz1_vrd_lai.txt')

#Load the initial states and uncontrolled inputs (i.e. ET and Crop Coeff)
initialState_mz1 = np.loadtxt("./common/initial_states/Init_state_rz_mz1_vrd_lai.txt")
initialIrrigation_mz1 = np.loadtxt("./common/initial_states/Init_irrig_mz1_vrd_lai.txt")
initialIrrigation_mz1 = initialIrrigation_mz1*86400


referenceEvap_all_mz1 = _ref_Evap_predicted/(86400*1000)
referenceEvap_all_act_mz1 = _ref_Evap/(86400*1000)
cropCoefficient_all_mz1 = _crop_coeff
rain_all_predicted = -1*(_rain_predicted/1000)
rain_all_act = -1*(_rain_act/1000)
lai_factors_all = np.ones(180)

#Pressure head values for the entire soil column, for the 3 MZs
allHeadValues_mz1 = np.loadtxt("./common/initial_states/Init_state_all_mz1_vrd_lai.txt")

#Relevant for the simulation
sequenceLength = 5 
# horizonLength_orig = 14
horizonLength_orig = 7
horizonLength_weather = int(sequenceLength + horizonLength_orig)

simulationPeriod =  123 #Can be tuned
rooting_depths = np.zeros(160)
rooting_depths[0:72] = 0.50 
rooting_depths[72:] = 0.50

#Lists to store the simulation results
statesList_mz1 = []
irrigationList_mz1 = []


for j in range(sequenceLength-1):
    statesList_mz1+=[initialState_mz1[j]]
    pass

statesList_mz1+=[compute_root_zone_moisture(obtain_soil_moisture(1*allHeadValues_mz1,soilPars_mz1),0.50)]

for j in range(sequenceLength-1):
    irrigationList_mz1+=[initialIrrigation_mz1[j]]
    pass
 

#Lower and upper bounds of the target zone for each MZ
LowerZone_mz1 = 0.200
UpperZone_mz1 = 0.280 #0.280 

k=0
x0_init_mz1 = np.array(statesList_mz1[k : k+sequenceLength])
u_init_mz1  = np.array(irrigationList_mz1[k : k+sequenceLength-1])
kc_init_mz1 = cropCoefficient_all_mz1[k : k+horizonLength_weather]
et_init_mz1 = referenceEvap_all_mz1[k : k+horizonLength_weather]

et_init_mz1_act = referenceEvap_all_act_mz1[k : k+horizonLength_weather]
rain = rain_all_predicted[k : k+horizonLength_weather]
rain_act = rain_all_act[k : k+horizonLength_weather]
rooting_depth = rooting_depths[k:k+horizonLength_weather]
lai_factors = lai_factors_all[k:k+horizonLength_weather]

x0_scaled_mz1 = (x0_init_mz1 - data_min_mz1[0])/(data_max_mz1[0]-data_min_mz1[0])
kc_scaled_mz1 = (kc_init_mz1 - data_min_mz1[2])/(data_max_mz1[2]-data_min_mz1[2])
et_scaled_mz1 = (et_init_mz1 - data_min_mz1[3])/(data_max_mz1[3]-data_min_mz1[3])
rd_scaled_mzs = (rooting_depth - data_min_mz1[4])/(data_max_mz1[4]-data_min_mz1[4])
lai_scaled_mzs = (lai_factors - data_min_mz1[5])/(data_max_mz1[5]-data_min_mz1[5])


lbd = -10e18
ubd = 10e18

# guessesX = x0_scaled_mz1[0]*np.ones(horizonLength_orig)

guessesX = np.array([0.224,0.252,0.246,0.242,0.234,0.232,0.229,
                    0.226,0.224,0.221,0.219,0.216,0.213,0.210])

guessesX = (guessesX - data_min_mz1[0])/(data_max_mz1[0]-data_min_mz1[0])
guessesX = guessesX.tolist()

guessesU = -0.003887*np.ones(horizonLength_orig) # not scaled

guessesC = np.ones(horizonLength_orig)

x,u,c,ubd,lbd=run_gbd_for_scheduler(x0_scaled_mz1,u_init_mz1,guessesX,guessesU, 
                guessesC,kc_scaled_mz1,et_scaled_mz1,rd_scaled_mzs,lai_scaled_mzs,rain,ubd,lbd)
