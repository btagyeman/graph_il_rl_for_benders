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

from core_agent.environments import GBDENV
from core_agent.memory import PPOMemory
from core_agent.ppo_agent import PPOAgent, compute_gae
from parameter_verification.generate_data import EvaluateParameterSet


def _ensure_repo_src_on_path() -> None:
    current = Path(__file__).resolve()
    for parent in current.parents:
        candidate = parent / "src"
        if (candidate / "graph_il_rl_for_benders").exists():
            if str(candidate) not in sys.path:
                sys.path.insert(0, str(candidate))
            return


_ensure_repo_src_on_path()

from graph_il_rl_for_benders.config_schemas import validate_cs1_train_rl_config


@dataclass
class TrainConfig:
    parameter_file: Path
    output_dir: Path
    episodes: Optional[int]
    batch_size: int
    max_steps_per_episode: int
    save_every: int
    seed: int
    enable_live_plot: bool


def modify_action_and_probs(action_old, modified_action, probs, log_probs):
    idx_uncertain = []
    for idx in range(len(action_old)):
        if action_old[idx] == -1:
            idx_uncertain.append(idx)

    if len(idx_uncertain) > 0:
        for idx in range(len(idx_uncertain)):
            replaced_action = modified_action[idx]
            if replaced_action == 1:
                probs[idx] = 1 - 1e-6
                log_probs[idx] = np.log(1 - 1e-6)
            else:
                probs[idx] = 1e-6
                log_probs[idx] = np.log(1e-6)

    return probs, log_probs


def run_training(config: TrainConfig):
    np.random.seed(config.seed)
    tf.random.set_seed(config.seed)

    generated_parameters = np.loadtxt(config.parameter_file)
    episodes = len(generated_parameters) if config.episodes is None else config.episodes
    if episodes > len(generated_parameters):
        raise ValueError(
            f"episodes={episodes} exceeds available parameter sets={len(generated_parameters)} in {config.parameter_file}."
        )

    saved_agents_dir = config.output_dir / "saved_agents"
    os.makedirs(saved_agents_dir, exist_ok=True)

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

    for ep in range(episodes):
        initial_guesses = [
            [0, 1, 1, 1, 0],
            [0, 1, 0, 1, 0],
            [0, 1, 0, 0, 0],
            [0, 1, 1, 0, 0],
            [1, 0, 1, 1, 0],
            [1, 0, 0, 1, 0],
            [1, 0, 0, 0, 0],
            [1, 0, 1, 0, 0],
        ]
        x_guess = [1, 1, 1, 1, 1, 1]
        y_guess = initial_guesses[np.random.randint(len(initial_guesses))]

        ep_eval = EvaluateParameterSet(initial_guess=y_guess, par_dict=generated_parameters[ep])
        ep_eval.evaluate_parameter_set()
        if not ep_eval.use_parameters:
            print(f"Episode {ep + 1}/{episodes} | Skipping training for unsuitable parameters.")
            continue

        env = GBDENV(
            UBD=1e5,
            LBD=-1e5,
            feas_pars=generated_parameters[ep],
            x_guess=x_guess,
            y_guess=y_guess,
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
            modify_action_and_probs(action, modified_action, probs, log_prob)
            rewards_ep.append(reward)
            memory.store_transitions(state, action, log_prob, reward, done, value)

            state = next_state
            values.append(float(value))
            episode_reward += reward
            step += 1

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

        print(f"Episode {ep + 1}/{episodes} | Reward: {episode_reward:.2f} | Avg (last 100): {average_reward:.2f}")

        if (ep + 1) % config.save_every == 0:
            checkpoint = int((ep + 1) / config.save_every)
            print(f"Saving model after {config.save_every} episodes...")
            agent.actor.save(saved_agents_dir / f"rl_agent_logits_il_{checkpoint}_episodes")
            np.savetxt(config.output_dir / f"average_rewards_il_{checkpoint}_episodes.txt", np.array(average_rewards))
            np.savetxt(config.output_dir / f"episodic_rewards_il_{checkpoint}_episodes.txt", np.array(episodic_rewards))

    if config.enable_live_plot:
        plt.ioff()
        plt.show()

    np.savetxt(config.output_dir / "average_rewards_il.txt", np.array(average_rewards))
    np.savetxt(config.output_dir / "episodic_rewards_il.txt", np.array(episodic_rewards))
    agent.actor.save(saved_agents_dir / "rl_agent_il")


def _load_config(config_path: Optional[Path]) -> dict:
    if config_path is None:
        return {}
    with config_path.open("r", encoding="utf-8") as file:
        payload = json.load(file)
    if not isinstance(payload, dict):
        raise ValueError("Config file must contain a JSON object.")
    validate_cs1_train_rl_config(payload)
    return payload


def parse_args(argv=None) -> TrainConfig:
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", type=Path, default=None)
    pre_args, remaining = pre_parser.parse_known_args(argv)
    config_values = _load_config(pre_args.config)

    parser = argparse.ArgumentParser(description="Train RL agent for Case Study 1.")
    parser.add_argument("--config", type=Path, default=pre_args.config)
    parser.add_argument(
        "--parameter-file",
        type=Path,
        default=Path(config_values.get("parameter_file", "./parameter_verification/generated_parameters.txt")),
    )
    parser.add_argument("--output-dir", type=Path, default=Path(config_values.get("output_dir", "./results")))
    default_episodes = config_values.get("episodes", None)
    parser.add_argument("--episodes", type=int, default=None if default_episodes is None else int(default_episodes))
    parser.add_argument("--batch-size", type=int, default=int(config_values.get("batch_size", 32)))
    parser.add_argument("--max-steps-per-episode", type=int, default=int(config_values.get("max_steps_per_episode", 1000)))
    parser.add_argument("--save-every", type=int, default=int(config_values.get("save_every", 500)))
    parser.add_argument("--seed", type=int, default=int(config_values.get("seed", 42)))
    default_no_plot = bool(config_values.get("no_live_plot", False))
    parser.add_argument("--no-live-plot", action="store_true", default=default_no_plot)

    args = parser.parse_args(remaining)
    return TrainConfig(
        parameter_file=args.parameter_file,
        output_dir=args.output_dir,
        episodes=args.episodes,
        batch_size=args.batch_size,
        max_steps_per_episode=args.max_steps_per_episode,
        save_every=args.save_every,
        seed=args.seed,
        enable_live_plot=not args.no_live_plot,
    )


def main():
    config = parse_args()
    run_training(config)


if __name__ == "__main__":
    main()
