import numpy as np 

def evaluate_Kc_hard_spring_wheat(x):
    return -0.01925 + 0.002641*x + (1.05e-07)*(x**2) - (2.23e-09)*(x**3) + (3.57e-13)*(x**4)

def evaluate_Kc_soft_spring_wheat(x):
    return -0.0207 + 0.00266*x + (4.7e-08)*(x**2) - (2.0e-09)*(x**3) + (2.7e-13)*(x**4)

#Evaluates the crop coefficient using the daily average temperatures and the cummulative growing degree days (GDD)
def generate_Kc(daily_average_temps):
    for i in range(len(daily_average_temps)):
        if daily_average_temps[i] > 25.0:
            daily_average_temps[i] = 25.0
            pass 
        pass
    base_temp = 5 
    gdd = daily_average_temps - base_temp
    gdd_cum = np.cumsum(gdd)
    kc_calculated = evaluate_Kc_soft_spring_wheat(gdd_cum)
    for i in range(len(kc_calculated)):
        if kc_calculated[i]<0.20:
            kc_calculated[i] = 0.20
        if kc_calculated[i] > 1.25:
            kc_calculated[i] = 1.25
    return kc_calculated
