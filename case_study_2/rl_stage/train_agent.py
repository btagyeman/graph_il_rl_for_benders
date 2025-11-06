import tensorflow as tf
import numpy as np
from core_agent.ppo_agent import PPOAgent, compute_gae
from core_agent.memory import PPOMemory
from core_agent.environments import GBD_MIMPC
from common.actual_field.temporal_spatial_parameters_field import *
from common.actual_field.richards_equation_field import volMoistureAllNodes as vwc_fun
from common.utils_season_long import compute_root_zone_moisture,obtain_soil_moisture
from matplotlib import pyplot as plt
import os
from common.utils_season_long import evaluate_weather_uncertainy_ET,evaluate_weather_uncertainty_rain

os.makedirs("./results", exist_ok=True)
os.makedirs("./results/saved_agents", exist_ok=True)



soilPars_mz1 = pars_mz1()


#load the weather and crop information 
avg_temp_all = np.loadtxt('./common/weather_data/davg_temp_all.txt')
kc_all = np.loadtxt('./common/weather_data/kc_all.txt')
rain_all = np.loadtxt('./common/weather_data/drain_all.txt')
rain_all = -rain_all/1000  # Convert to mm/day
ref_evap_all = np.loadtxt('./common/weather_data/dpet_all.txt')
ref_evap_all = ref_evap_all/(86400*1000)
lai_all = np.ones(len(avg_temp_all))  
rds_all = 0.50*np.ones(len(avg_temp_all)) 

#load the initial state and irrigation amounts
init_state = np.loadtxt("./common/initial_states/Init_state_rz_mz1_vrd_lai.txt")
init_irrig = np.loadtxt("./common/initial_states/Init_irrig_mz1_vrd_lai.txt")
init_irrig = init_irrig*86400
all_head_vals = np.loadtxt("./common/initial_states/Init_state_all_mz1_vrd_lai.txt")

seq_length = 5
pred_horizon = 14
horizon_weather = pred_horizon + seq_length 

states_list = []
irrig_list = []
for i in range(seq_length-1):
    states_list.append(init_state[i])
    irrig_list.append(init_irrig[i])
    pass 
states_list+=[compute_root_zone_moisture(obtain_soil_moisture(1.05*all_head_vals,soilPars_mz1,vwc_fun),0.50)]

all_states = []
all_states.append(all_head_vals)



data_min = np.loadtxt('./trained_lstm_models/model_weights/data_min_mz1_vrd_lai.txt')
data_max = np.loadtxt('./trained_lstm_models/model_weights/data_max_mz1_vrd_lai.txt')

#Define the guesses
guessesX=np.array([0.224,0.252,0.246,0.242,0.234,0.232,0.229,0.226,0.224,0.221,0.219,0.216,0.213,0.210])
guessesX = (guessesX - data_min[0]) / (data_max[0] - data_min[0])  # Scale guesses
guessesU=-0.003887*np.ones(len(guessesX))
guessesC=np.ones(len(guessesX)) 
lbd = -1e6
ubd = 1e6

# the tuning matrices for the estimator, not needed for this agent 
meas_cov = 19.25 
proc_cov = 0.05*np.eye(21)
cov = 15.9*np.ones((21,21))


# Main training settings
episodes = 5000
agent = PPOAgent()
memory = PPOMemory()
batch_size = 32
max_steps_per_episode = 50
epsilon = 0.01

# Logging
episodic_rewards = []
average_rewards = []

# Plotting setup
plt.ion()
fig, ax = plt.subplots()
reward_line, = ax.plot([], [], label='Average Reward (last 100)')
ax.set_xlabel('Episode')
ax.set_ylabel('Average Reward')
ax.set_title('PPO Training Progress')
ax.legend()


# Training loop
for ep in range(episodes):

    x0_ep = np.array(states_list[ep:ep+seq_length])
    u_ep  = np.array(irrig_list[ep:ep+seq_length-1])
    kc_ep = kc_all[ep:ep+horizon_weather]
    et_ep = ref_evap_all[ep:ep+horizon_weather]

    et_ep_nsy = evaluate_weather_uncertainy_ET(et_ep,len(et_ep))
    rain_ep=rain_all[ep:ep+horizon_weather]
    rd_ep=rds_all[ep:ep+horizon_weather]
    rain_ep_nsy=evaluate_weather_uncertainty_rain(rain_ep,len(rain_ep))
    lai_ep = lai_all[ep:ep+horizon_weather]

    x0_ep_scld=(x0_ep-data_min[0])/(data_max[0]-data_min[0])
    kc_ep_scld=(kc_ep-data_min[2])/(data_max[2]-data_min[2])
    et_ep_scld=(et_ep_nsy-data_min[3])/(data_max[3]-data_min[3])
    rd_ep_scld=(rd_ep-data_min[4])/(data_max[4]-data_min[4])
    lai_ep_scld=(lai_ep-data_min[5])/(data_max[5]-data_min[5])

    env = GBD_MIMPC(x0_ep_scld,u_ep,guessesX,guessesU,guessesC,kc_ep_scld,et_ep_scld,rd_ep_scld,lai_ep_scld,rain_ep_nsy,lbd,ubd,epsilon)
    state = env.reset()
    # print(state)
    done = False
    episode_reward = 0
    values = []
    rewards_ep = []
    # episodic_rewards_performance = []
    step = 0

    while not done and step < max_steps_per_episode:
        # print(done)
        action, log_prob, value, probs = agent.choose_action(state)
        next_state, reward, done, _, modified_action = env.step(action)
        # probs_mod, log_probs_mod = modify_action_and_probs(action, modified_action, probs, log_prob)
        rewards_ep.append(reward)
        # episodic_rewards_norm.append(reward)
        memory.store_transitions(state, action, log_prob, reward, done, value)
        # memory.store_transitions(state, np.array(modified_action), np.array(log_probs_mod), reward, done, value)

        state = next_state
        # values.append(value)
        # values.extend([value])
        # values.append(value.item())  # ensure it's a scalar float
        values.append(float(value))  # Force cast to pure float
        episode_reward += reward
        step += 1

    u_all, x_all, c_all, x_head_all, vol_all, rz_vol_moist = env.obtain_next_vwc(all_head_vals,et_ep[4],rain_ep[4], kc_ep[4])

    guessesC = c_all 
    guessesX = (x_all - data_min[0])/(data_max[0] - data_min[0])  # Scale guesses
    guessesU = u_all 

    irrig_list.append(u_all[0]) 
    states_list.append(rz_vol_moist)
    all_states.append(x_head_all)

    x_head_vals = x_head_all

    rewards_ep = np.array(rewards_ep)/len(rewards_ep)
    # rewards_ep = np.array(rewards_ep)/1
    memory.store_rewards(list(rewards_ep))

    bootstrap_value = 0.0 if done else agent.critic(state).numpy().squeeze()
    next_values = values[1:] + [bootstrap_value]
    memory.store_next_values(next_values)

    # Compute learning target
    states, actions, log_probs, rewards, dones, values, next_values = memory.get_trajectories()
    # print(len(states))
    # if len(states) >= batch_size:

    if len(actions) >= batch_size:
        print("Training agent...")
        advantages = compute_gae(rewards, values, next_values, dones, agent.gamma, agent.lam)
        advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-8)  # Normalize
        returns = tf.convert_to_tensor(np.array(advantages) + np.array(values), dtype=tf.float32)
        agent.learn(memory, advantages, returns)
        memory.clear()
    episode_reward = episode_reward/len(rewards_ep)
    episodic_rewards.append(episode_reward)
    average_reward = np.mean(episodic_rewards[-100:])
    average_rewards.append(round(average_reward,3))

    # Live update plot
    reward_line.set_data(range(len(average_rewards)), average_rewards)
    ax.relim()
    ax.autoscale_view()
    plt.pause(0.01)

    print(f"Episode {ep + 1}/{episodes} | Reward: {episode_reward:.2f} | Avg (last 100): {average_reward:.2f}")

    if (ep+1) % 100 == 0: 
        k = int((ep + 1) / 100)
        print("Saving results after 500 episodes...")
        np.savetxt(f"./results/average_rewards_{k}.txt", np.array(average_rewards))
        np.savetxt(f"./results/episodic_rewards_{k}.txt", np.array(episodic_rewards))
        agent.actor.save(f"./results/saved_agents/rl_agent_{k}")

plt.ioff()
plt.show()

np.savetxt("./results/average_rewards.txt", np.array(average_rewards))
np.savetxt("./results/episodic_rewards.txt", np.array(episodic_rewards))
agent.actor.save("./results/saved_agents/rl_agent")
