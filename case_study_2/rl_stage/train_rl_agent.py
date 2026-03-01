from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from common.actual_field.richards_equation_field import volMoistureAllNodes as vwc_fun
from common.actual_field.temporal_spatial_parameters_field import pars_mz1
from common.utils_season_long import (
    compute_root_zone_moisture,
    evaluate_weather_uncertainy_ET,
    evaluate_weather_uncertainty_rain,
    obtain_soil_moisture,
)
from core_agent.environments import GBD_MIMPC
from core_agent.memory import PPOMemory
from core_agent.ppo_agent import PPOAgent, compute_gae


def _ensure_repo_src_on_path() -> None:
    current = Path(__file__).resolve()
    for parent in current.parents:
        candidate = parent / "src"
        if (candidate / "graph_il_rl_for_benders").exists():
            if str(candidate) not in sys.path:
                sys.path.insert(0, str(candidate))
            return


_ensure_repo_src_on_path()

from graph_il_rl_for_benders.config_schemas import validate_cs2_train_rl_config


@dataclass
class TrainConfig:
    common_dir: Path
    model_weights_dir: Path
    output_dir: Path
    episodes: int
    batch_size: int
    max_steps_per_episode: int
    epsilon: float
    save_every: int
    seed: int
    enable_live_plot: bool


def run_training(config: TrainConfig):
    np.random.seed(config.seed)
    tf.random.set_seed(config.seed)

    os.makedirs(config.output_dir, exist_ok=True)
    os.makedirs(config.output_dir / "saved_agents", exist_ok=True)

    soil_pars_mz1 = pars_mz1()

    weather_dir = config.common_dir / "weather_data"
    initial_states_dir = config.common_dir / "initial_states"

    avg_temp_all = np.loadtxt(weather_dir / "davg_temp_all.txt")
    kc_all = np.loadtxt(weather_dir / "kc_all.txt")
    rain_all = np.loadtxt(weather_dir / "drain_all.txt")
    rain_all = -rain_all / 1000
    ref_evap_all = np.loadtxt(weather_dir / "dpet_all.txt")
    ref_evap_all = ref_evap_all / (86400 * 1000)
    lai_all = np.ones(len(avg_temp_all))
    rds_all = 0.50 * np.ones(len(avg_temp_all))

    init_state = np.loadtxt(initial_states_dir / "Init_state_rz_mz1_vrd_lai.txt")
    init_irrig = np.loadtxt(initial_states_dir / "Init_irrig_mz1_vrd_lai.txt")
    init_irrig = init_irrig * 86400
    all_head_vals = np.loadtxt(initial_states_dir / "Init_state_all_mz1_vrd_lai.txt")

    seq_length = 5
    pred_horizon = 14
    horizon_weather = pred_horizon + seq_length

    states_list = []
    irrig_list = []
    for idx in range(seq_length - 1):
        states_list.append(init_state[idx])
        irrig_list.append(init_irrig[idx])
    states_list += [
        compute_root_zone_moisture(
            obtain_soil_moisture(1.05 * all_head_vals, soil_pars_mz1, vwc_fun),
            0.50,
        )
    ]

    all_states = [all_head_vals]

    data_min = np.loadtxt(config.model_weights_dir / "data_min_mz1_vrd_lai.txt")
    data_max = np.loadtxt(config.model_weights_dir / "data_max_mz1_vrd_lai.txt")

    guesses_x = np.array([0.224, 0.252, 0.246, 0.242, 0.234, 0.232, 0.229, 0.226, 0.224, 0.221, 0.219, 0.216, 0.213, 0.210])
    guesses_x = (guesses_x - data_min[0]) / (data_max[0] - data_min[0])
    guesses_u = -0.003887 * np.ones(len(guesses_x))
    guesses_c = np.ones(len(guesses_x))
    lbd = -1e6
    ubd = 1e6

    agent = PPOAgent()
    memory = PPOMemory()

    episodic_rewards = []
    average_rewards = []

    if config.enable_live_plot:
        plt.ion()
        fig, ax = plt.subplots()
        reward_line, = ax.plot([], [], label="Average Reward (last 100)")
        ax.set_xlabel("Episode")
        ax.set_ylabel("Average Reward")
        ax.set_title("PPO Training Progress")
        ax.legend()

    for ep in range(config.episodes):
        x0_ep = np.array(states_list[ep : ep + seq_length])
        u_ep = np.array(irrig_list[ep : ep + seq_length - 1])
        kc_ep = kc_all[ep : ep + horizon_weather]
        et_ep = ref_evap_all[ep : ep + horizon_weather]

        et_ep_nsy = evaluate_weather_uncertainy_ET(et_ep, len(et_ep))
        rain_ep = rain_all[ep : ep + horizon_weather]
        rd_ep = rds_all[ep : ep + horizon_weather]
        rain_ep_nsy = evaluate_weather_uncertainty_rain(rain_ep, len(rain_ep))
        lai_ep = lai_all[ep : ep + horizon_weather]

        x0_ep_scld = (x0_ep - data_min[0]) / (data_max[0] - data_min[0])
        kc_ep_scld = (kc_ep - data_min[2]) / (data_max[2] - data_min[2])
        et_ep_scld = (et_ep_nsy - data_min[3]) / (data_max[3] - data_min[3])
        rd_ep_scld = (rd_ep - data_min[4]) / (data_max[4] - data_min[4])
        lai_ep_scld = (lai_ep - data_min[5]) / (data_max[5] - data_min[5])

        env = GBD_MIMPC(
            x0_ep_scld,
            u_ep,
            guesses_x,
            guesses_u,
            guesses_c,
            kc_ep_scld,
            et_ep_scld,
            rd_ep_scld,
            lai_ep_scld,
            rain_ep_nsy,
            lbd,
            ubd,
            config.epsilon,
        )
        state = env.reset()
        done = False
        episode_reward = 0.0
        values = []
        rewards_ep = []
        step = 0

        while not done and step < config.max_steps_per_episode:
            action, log_prob, value, probs = agent.choose_action(state)
            next_state, reward, done, _, modified_action = env.step(action)
            rewards_ep.append(reward)
            memory.store_transitions(state, action, log_prob, reward, done, value)

            state = next_state
            values.append(float(value))
            episode_reward += reward
            step += 1

        u_all, x_all, c_all, x_head_all, vol_all, rz_vol_moist = env.obtain_next_vwc(
            all_head_vals,
            et_ep[4],
            rain_ep[4],
            kc_ep[4],
        )

        guesses_c = c_all
        guesses_x = (x_all - data_min[0]) / (data_max[0] - data_min[0])
        guesses_u = u_all

        irrig_list.append(u_all[0])
        states_list.append(rz_vol_moist)
        all_states.append(x_head_all)

        rewards_ep = np.array(rewards_ep) / len(rewards_ep)
        memory.store_rewards(list(rewards_ep))

        bootstrap_value = 0.0 if done else agent.critic(state).numpy().squeeze()
        next_values = values[1:] + [bootstrap_value]
        memory.store_next_values(next_values)

        _, actions, _, rewards, dones, values, next_values = memory.get_trajectories()
        if len(actions) >= config.batch_size:
            print("Training agent...")
            advantages = compute_gae(rewards, values, next_values, dones, agent.gamma, agent.lam)
            advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-8)
            returns = tf.convert_to_tensor(np.array(advantages) + np.array(values), dtype=tf.float32)
            agent.learn(memory, advantages, returns)
            memory.clear()

        episode_reward = episode_reward / len(rewards_ep)
        episodic_rewards.append(episode_reward)
        average_reward = np.mean(episodic_rewards[-100:])
        average_rewards.append(round(average_reward, 3))

        if config.enable_live_plot:
            reward_line.set_data(range(len(average_rewards)), average_rewards)
            ax.relim()
            ax.autoscale_view()
            plt.pause(0.01)

        print(f"Episode {ep + 1}/{config.episodes} | Reward: {episode_reward:.2f} | Avg (last 100): {average_reward:.2f}")

        if (ep + 1) % config.save_every == 0:
            checkpoint = int((ep + 1) / config.save_every)
            print(f"Saving results after {config.save_every} episodes...")
            np.savetxt(config.output_dir / f"average_rewards_{checkpoint}.txt", np.array(average_rewards))
            np.savetxt(config.output_dir / f"episodic_rewards_{checkpoint}.txt", np.array(episodic_rewards))
            agent.actor.save(config.output_dir / "saved_agents" / f"rl_agent_{checkpoint}")

    if config.enable_live_plot:
        plt.ioff()
        plt.show()

    np.savetxt(config.output_dir / "average_rewards.txt", np.array(average_rewards))
    np.savetxt(config.output_dir / "episodic_rewards.txt", np.array(episodic_rewards))
    agent.actor.save(config.output_dir / "saved_agents" / "rl_agent")


def _load_config(config_path: Optional[Path]) -> dict:
    if config_path is None:
        return {}
    with config_path.open("r", encoding="utf-8") as file:
        payload = json.load(file)
    if not isinstance(payload, dict):
        raise ValueError("Config file must contain a JSON object.")
    validate_cs2_train_rl_config(payload)
    return payload


def parse_args(argv=None) -> TrainConfig:
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", type=Path, default=None)
    pre_args, remaining = pre_parser.parse_known_args(argv)
    config_values = _load_config(pre_args.config)

    parser = argparse.ArgumentParser(description="Train RL agent for Case Study 2.")
    parser.add_argument("--config", type=Path, default=pre_args.config)
    parser.add_argument("--common-dir", type=Path, default=Path(config_values.get("common_dir", "./common")))
    parser.add_argument(
        "--model-weights-dir",
        type=Path,
        default=Path(config_values.get("model_weights_dir", "./trained_lstm_models/model_weights")),
    )
    parser.add_argument("--output-dir", type=Path, default=Path(config_values.get("output_dir", "./results")))
    parser.add_argument("--episodes", type=int, default=int(config_values.get("episodes", 5000)))
    parser.add_argument("--batch-size", type=int, default=int(config_values.get("batch_size", 32)))
    parser.add_argument("--max-steps-per-episode", type=int, default=int(config_values.get("max_steps_per_episode", 50)))
    parser.add_argument("--epsilon", type=float, default=float(config_values.get("epsilon", 0.01)))
    parser.add_argument("--save-every", type=int, default=int(config_values.get("save_every", 100)))
    parser.add_argument("--seed", type=int, default=int(config_values.get("seed", 42)))
    default_no_plot = bool(config_values.get("no_live_plot", False))
    parser.add_argument("--no-live-plot", action="store_true", default=default_no_plot)

    args = parser.parse_args(remaining)
    return TrainConfig(
        common_dir=args.common_dir,
        model_weights_dir=args.model_weights_dir,
        output_dir=args.output_dir,
        episodes=args.episodes,
        batch_size=args.batch_size,
        max_steps_per_episode=args.max_steps_per_episode,
        epsilon=args.epsilon,
        save_every=args.save_every,
        seed=args.seed,
        enable_live_plot=not args.no_live_plot,
    )


def main():
    config = parse_args()
    run_training(config)


if __name__ == "__main__":
    main()
