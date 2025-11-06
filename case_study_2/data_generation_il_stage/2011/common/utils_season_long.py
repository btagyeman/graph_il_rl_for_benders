import numpy as np 

#Some user defined functions that introduce uncertainty in the weather conditions
def evaluate_weather_uncertainy_ET(ref_Evap_sequence, prediction_horizon):
    noise = []
    for i in range(prediction_horizon):
        _std_dev = (i+1)*0.02*1e-09
        np.random.seed(i)
        _noise = np.random.normal(0, _std_dev)
        noise.append(_noise)
        pass 
    ref_Evap_sequence_noisy = ref_Evap_sequence*(86400*1000) + np.array(noise)*(86400*1000)
    for i in range(len(ref_Evap_sequence_noisy)):
        if ref_Evap_sequence_noisy[i]< 1.04:
            ref_Evap_sequence_noisy[i] = 1.04
        if ref_Evap_sequence_noisy[i] > 9.0:
            ref_Evap_sequence_noisy[i] = 9.0
    pass
    return ref_Evap_sequence_noisy/(86400*1000)

def evaluate_weather_uncertainty_rain(rain_sequence, prediction_horizon):
    noise = []
    for i in range(prediction_horizon):
        _std_dev = (i+1)/14.0 #previous value, 7.0
        np.random.seed(i)
        _noise = np.random.normal(0, _std_dev)
        noise.append(_noise)
        pass 
    rain_sequence_noisy = np.zeros(len(rain_sequence))
    noise = np.array(noise)

    for i in range(prediction_horizon):
        if rain_sequence[i] == 0:
            rain_sequence_noisy[i]=0 
        else:
            rain_sequence_noisy[i] = -1*rain_sequence[i]*1000 + noise[i]
    rain_sequence_noisy = np.abs(rain_sequence_noisy)
    rain_sequence_noisy[rain_sequence_noisy<0.5] = 0.0
    return -rain_sequence_noisy/1000

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
def obtain_soil_moisture(head, soilPars, vwc_fun):
    vol_moist = vwc_fun(head, soilPars)
    return vol_moist