import tensorflow as tf
import numpy as np
from helper_functions.generate_problem_instances import GenerateDataset
from parameter_verification.generate_data import EvaluateParameterSet
from core_agent.ppo_agent import PPOAgent, compute_gae
from core_agent.memory import PPOMemory
from core_agent.environments import GBDENV
from matplotlib import pyplot as plt
import os
# # Generate the feasible parameters for training
# test_parameters = GenerateDataset(n_samples=1000)
# test_parameters.generate_feasible_parameters()
# generated_parameters = test_parameters.feasible_parameters

# # It will make sense to save the generated parameters for later use
# np.savetxt("./results/generated_parameters.txt", np.array(generated_parameters))


os.makedirs("./results/saved_agents/", exist_ok=True)
# Load the saved parameters
generated_parameters = np.loadtxt("./parameter_verification/generated_parameters.txt")

# Main training settings
episodes = len(generated_parameters)
agent = PPOAgent()
memory = PPOMemory()
batch_size = 32
max_steps_per_episode = 1000

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

# def modify_action_and_probs(action_old, modified_action, probs, log_probs):
#     """
#     Replace uncertain actions in probs/log_probs using solver's actual assignment.
    
#     Parameters:
#     - action_old: original action with -1 for uncertain entries
#     - modified_action: full binary action (agent + solver)
#     - probs: predicted probabilities from the agent (already clipped)
#     - log_probs: initial log_probs (use placeholder -1 for uncertain ones)
    
#     Returns:
#     - updated probs and log_probs with solver-based log_probs for uncertain entries
#     """
#     for i in range(len(action_old)):
#         if action_old[i] == -1:
#             p = np.clip(probs[i], 1e-6, 1 - 1e-6)
#             log_probs[i] = np.log(p) if modified_action[i] == 1 else np.log(1 - p)
    
#     return probs, log_probs

def modify_action_and_probs(action_old, modified_action, probs,log_probs):
    idx_uncertain = []
    probs = probs 
    log_probs = log_probs

    for i in range(len(action_old)):
        if action_old[i] == -1:
            idx_uncertain.append(i)
            pass 
        pass 

    if len(idx_uncertain) > 0:
        for i in range(len(idx_uncertain)):
            replaced_action = modified_action[i]
            if replaced_action ==1:
                probs[i] =  1- 1e-6
                log_probs[i] = np.log(1- 1e-6)
            else:
                probs[i] = 1e-6
                log_probs[i] = np.log(1e-6)
    else:
        pass
                
    return probs, log_probs

# Training loop
# for ep in range(episodes):
for ep in range(50000):
    # Random initial guess
    # initial_guesses = [[y1, y2, y3] for y1 in [0, 1] for y2 in [0, 1] for y3 in [0, 1]] # add some variety in the initialization of the GBD
    # initial_guesses =[[1,0,0,0,0], [0,1,0,0,0]]
    initial_guesses = [[0,1,1,1,0], [0,1,0,1,0],[0,1,0,0,0],[0,1,1,0,0], [1,0,1,1,0],[1,0,0,1,0], [1,0,0,0,0], [1,0,1,0,0]]
    x_guess = [1,1,1,1,1,1]
    # initial_guesses = [[0, 1, 0]]
    y_guess = initial_guesses[np.random.randint(len(initial_guesses))]


    #Check if the parameters are suitable for training
    # ep_parameters = generated_parameters[ep]
    ep_eval = EvaluateParameterSet(initial_guess=y_guess, par_dict=generated_parameters[ep])
    ep_eval.evaluate_parameter_set()

    if not ep_eval.use_parameters:
        print(f"Episode {ep + 1}/{episodes} | Skipping training for unsuitable parameters.")
        continue



    env = GBDENV(UBD=1e5, LBD=-1e5, feas_pars=generated_parameters[ep], x_guess=x_guess, y_guess=y_guess)
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
        # print(action)
        next_state, reward, done, _, modified_action = env.step(action)
        probs_mod, log_probs_mod = modify_action_and_probs(action, modified_action, probs, log_prob)
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

    if (ep + 1) % 500 == 0:
        k_ = int((ep + 1) / 500)
        print("Saving model after 500 episodes...")
        agent.actor.save(f"./results/saved_agents/rl_agent_logits_rand_{k_}_episodes")
        np.savetxt(f"./results/average_rewards_rand_{k_}_episodes.txt", np.array(average_rewards))
        np.savetxt(f"./results/episodic_rewards_rand_{k_}_episodes.txt", np.array(episodic_rewards))


plt.ioff()
plt.show()

np.savetxt("./results/average_rewards_randm.txt", np.array(average_rewards))
np.savetxt("./results/episodic_rewards_randm.txt", np.array(episodic_rewards))
agent.actor.save("./results/saved_agents/rl_agent_randm")
