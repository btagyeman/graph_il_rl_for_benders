from matplotlib import pyplot as plt
import numpy as np  

def GenerateTrajectories(episode_length, prescribed_irrigation, pressure_head,lower_zone, upper_zone):
    time_states = np.arange(episode_length+1) +1
    time_inputs = np.arange(episode_length)+1

    fig, axs = plt.subplots(2,1, figsize=(8,8))
    axs[0].plot(time_inputs, prescribed_irrigation*100, marker='.', linewidth=2)
    axs[0].set_title("Presecribed irrigation rate")
    axs[0].set_xlim(xmin=1)
    axs[0].set_ylabel("Irrigation rate (cm/day)")

    axs[1].plot(time_states, lower_zone*np.ones(episode_length+1), color='blue')
    axs[1].plot(time_states, upper_zone*np.ones(episode_length+1), color='blue')
    axs[1].plot(time_states, pressure_head, color='red')
    axs[1].set_xlim(xmin=1)
    axs[1].set_title('Root zone capillary pressure head')
    axs[1].set_ylabel('Pressure Head (m)')
    axs[1].set_xlabel('Time (days)')
    fig.tight_layout()
    plt.show()
    # plt.close()

    return 

def PlotRewardTrajectory(rewardTrajectory, filename):
    n_episodes = len(rewardTrajectory)
    n_episodes_array = np.arange(n_episodes) + 1 
    plt.figure(2, figsize=(9,9))
    plt.plot(n_episodes_array,rewardTrajectory)
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.tight_layout()
    plt.savefig("./plots/"+filename+".pdf")
    plt.show()
    # plt.close()
    return

def PlotRewardTrajectory_multiagent(rewardTrajectory_a, rewardTrajectory_b, rewardTrajectory_c, rewardTrajectory_d, filename):
    n_episodes = len(rewardTrajectory_a)
    n_episodes_array = np.arange(n_episodes) + 1 

    fig, axs = plt.subplots(4,1, figsize=(10,10))
    axs[0].plot(n_episodes_array, rewardTrajectory_a, color='red')
    axs[0].set_title("Reward trajectory - Upper agent")
    axs[0].set_xlim(xmin=1)
    #axs[0].set_xlabel("Time (days)")
    axs[0].set_ylabel("Reward")

    axs[1].plot(n_episodes_array, rewardTrajectory_b, color='blue')
    axs[1].set_title("Reward trajectory - Agent ($MZ_1$)")
    axs[1].set_xlim(xmin=1)
    #axs[1].set_xlabel("Episode")
    axs[1].set_ylabel("Reward")

    axs[2].plot(n_episodes_array, rewardTrajectory_c, color='green')
    axs[2].set_title("Reward trajectory - Agent ($MZ_2$)")
    axs[2].set_xlim(xmin=1)
    axs[2].set_xlabel("Episode")
    axs[2].set_ylabel("Reward")

    axs[3].plot(n_episodes_array, rewardTrajectory_d, color='green')
    axs[3].set_title("Reward trajectory - Agent ($MZ_3$)")
    axs[3].set_xlim(xmin=1)
    axs[3].set_xlabel("Episode")
    axs[3].set_ylabel("Reward")

    fig.tight_layout()
    plt.savefig("./plots/"+filename+".pdf")
    plt.show()
    # plt.close()
    return
