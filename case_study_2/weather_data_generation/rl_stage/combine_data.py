# -*- coding: utf-8 -*-
"""
Created on Sun Jun  8 19:55:52 2025

@author: agyem
"""

import numpy as np 

# Load the average temperature

avg_temp_edmonton = np.loadtxt("./combined_data/dAVG_TEMP_combined_edmonton.txt")
avg_temp_demo_farm = np.loadtxt("./combined_data/dAVG_TEMP_combined_demo_farm.txt")
avg_temp_leth_cda = np.loadtxt("./combined_data/dAVG_TEMP_combined_leth_cda.txt")
avg_temp_three_hills = np.loadtxt("./combined_data/dAVG_TEMP_combined_three_hills.txt")

avg_temp = np.concatenate([avg_temp_edmonton,avg_temp_demo_farm,avg_temp_leth_cda,avg_temp_three_hills])

kc_edmonton = np.loadtxt("./combined_data/kc_combined_edmonton.txt")
kc_demo_farm = np.loadtxt("./combined_data/kc_combined_demo_farm.txt")
kc_leth_cda = np.loadtxt("./combined_data/kc_combined_leth_cda.txt")
kc_three_hills = np.loadtxt("./combined_data/kc_combined_three_hills.txt")

kc = np.concatenate([kc_edmonton,kc_demo_farm,kc_leth_cda,kc_three_hills])

rain_edmonton = np.loadtxt("./combined_data/dRAIN_combined_edmonton.txt")
rain_demo_farm = np.loadtxt("./combined_data/dRAIN_combined_demo_farm.txt")
rain_leth_cda = np.loadtxt("./combined_data/dRAIN_combined_leth_cda.txt")
rain_three_hills = np.loadtxt("./combined_data/dRAIN_combined_three_hills.txt")

rain = np.concatenate([rain_edmonton,rain_demo_farm,rain_leth_cda,rain_three_hills])


pet_edmonton = np.loadtxt("./combined_data/dPET_combined_edmonton.txt")
pet_demo_farm = np.loadtxt("./combined_data/dPET_combined_demo_farm.txt")
pet_leth_cda = np.loadtxt("./combined_data/dPET_combined_leth_cda.txt")
pet_three_hills = np.loadtxt("./combined_data/dPET_combined_three_hills.txt")

pet = np.concatenate([pet_edmonton,pet_demo_farm,pet_leth_cda,pet_three_hills])

assert len(avg_temp) == len(kc) == len(rain) == len(pet)

# save the data

np.savetxt('./combined_data/kc_all.txt', kc)
np.savetxt('./combined_data/drain_all.txt', rain)
np.savetxt('./combined_data/dpet_all.txt', pet)
np.savetxt('./combined_data/davg_temp_all.txt', avg_temp)



