from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from spektral.data import Dataset
from spektral.data.loaders import DisjointLoader
from spektral.layers import ECCConv, GlobalSumPool
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense
from tqdm import tqdm


def _ensure_repo_src_on_path() -> None:
    current = Path(__file__).resolve()
    for parent in current.parents:
        candidate = parent / "src"
        if (candidate / "graph_il_rl_for_benders").exists():
            if str(candidate) not in sys.path:
                sys.path.insert(0, str(candidate))
            return


_ensure_repo_src_on_path()

from graph_il_rl_for_benders.config_schemas import validate_cs2_train_il_config


class GraphListDataset(Dataset):
    def __init__(self, graph_list, **kwargs):
        self.graph_list = graph_list
        super().__init__(**kwargs)

    def read(self):
        return self.graph_list


class ILAgent(Model):
    def __init__(self, num_outputs=14):
        super().__init__()
        self.conv1 = ECCConv(64, activation="relu")
        self.conv2 = ECCConv(64, activation="relu")
        self.conv3 = ECCConv(64, activation="relu")
        self.pool = GlobalSumPool()
        self.dense1 = Dense(64, activation="relu")
        self.dense2 = Dense(64, activation="relu")
        self.output_heads = [Dense(1, activation="sigmoid") for _ in range(num_outputs)]

    def call(self, inputs):
        x, a, e, i = inputs
        x = self.conv1([x, a, e])
        x = self.conv2([x, a, e])
        x = self.conv3([x, a, e])
        x = self.pool([x, i])
        x = self.dense1(x)
        x = self.dense2(x)
        outputs = [head(x) for head in self.output_heads]
        return tf.concat(outputs, axis=-1)


@dataclass
class TrainConfig:
    dataset_dir: Path
    output_dir: Path
    years: list[int]
    batch_size: int = 128
    epochs: int = 100
    train_split: float = 0.85
    threshold_high: float = 0.8
    threshold_low: float = 0.2
    seed: int = 42
    show_plots: bool = True


def elementwise_scale_node_feature(graph_dataset, tol=1e-20):
    all_vals = np.concatenate([graph.x.flatten() for graph in graph_dataset], axis=0)
    is_non_binary = ~(np.isclose(all_vals, 0.0, atol=tol) | np.isclose(all_vals, 1.0, atol=tol))

    non_binary_vals = all_vals[is_non_binary]
    if len(non_binary_vals) == 0:
        print("All values are binary (0 or 1). No scaling applied.")
        return graph_dataset, None, None

    x_mean = non_binary_vals.mean()
    x_std = non_binary_vals.std() + 1e-8

    for graph in graph_dataset:
        x_values = graph.x.flatten()
        x_scaled = []
        for value in x_values:
            if np.isclose(value, 0.0, atol=tol) or np.isclose(value, 1.0, atol=tol):
                x_scaled.append(value)
            else:
                x_scaled.append((value - x_mean) / x_std)
        graph.x = np.array(x_scaled, dtype=np.float32).reshape(-1, 1)

    return graph_dataset, x_mean, x_std


def load_yearly_datasets(dataset_dir: Path, years: list[int]):
    all_graphs = []
    for year in years:
        dataset_path = dataset_dir / f"graph_gbd_dataset_scheduler_{year}.pkl"
        with open(dataset_path, "rb") as dataset_file:
            all_graphs.extend(pickle.load(dataset_file))
    return all_graphs


def normalize_edge_features(graph_dataset):
    all_e = np.concatenate([graph.e for graph in graph_dataset], axis=0)
    e_mean = all_e.mean(axis=0)
    e_std = all_e.std(axis=0) + 1e-8

    for graph in graph_dataset:
        if graph.e is not None:
            graph.e = (graph.e - e_mean) / e_std

    return e_mean, e_std


def train_model(model, loader, loss_fn, optimizer):
    avg_train_losses = []
    epoch_loss = 0.0
    batches_in_epoch = 0
    current_epoch = 1

    total_steps = loader.steps_per_epoch * loader.epochs
    for batch in tqdm(loader, total=total_steps, desc="Training"):
        inputs, y_true = batch
        with tf.GradientTape() as tape:
            preds = model(inputs, training=True)
            loss = 0.0
            for idx in range(14):
                loss += loss_fn(y_true[:, idx], preds[:, idx])
            loss /= 14

        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        epoch_loss += float(loss.numpy())
        batches_in_epoch += 1

        if batches_in_epoch == loader.steps_per_epoch:
            avg_loss = epoch_loss / batches_in_epoch
            avg_train_losses.append(avg_loss)
            print(f"Epoch {current_epoch} - Avg Train Loss: {avg_loss:.4f}")
            current_epoch += 1
            epoch_loss = 0.0
            batches_in_epoch = 0

    return avg_train_losses


def evaluate_model(model, test_data):
    test_loader = DisjointLoader(GraphListDataset(test_data), batch_size=1, epochs=1, shuffle=True)

    @tf.function(experimental_relax_shapes=True)
    def predict(inputs):
        return model(inputs, training=False)

    @tf.function
    def compute_errors(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        return tf.reduce_mean(tf.abs(y_true - y_pred))

    y_preds = []
    y_trues = []
    total_mae = 0.0
    n_batches = 0

    pbar = tqdm(test_loader, total=test_loader.steps_per_epoch, desc="Evaluating")
    for batch in pbar:
        inputs, y_true = batch
        y_pred = predict(inputs)
        y_preds.append(y_pred.numpy().squeeze())
        y_trues.append(y_true.squeeze())
        mae = compute_errors(y_true, y_pred)
        total_mae += float(mae.numpy())
        n_batches += 1

    avg_mae = total_mae / max(1, n_batches)
    print(f"Test MAE: {avg_mae:.6f}")
    return np.array(y_preds), np.array(y_trues)


def run_training(config: TrainConfig):
    np.random.seed(config.seed)
    tf.random.set_seed(config.seed)

    os.makedirs(config.output_dir / "saved_agents", exist_ok=True)

    graph_dataset = load_yearly_datasets(config.dataset_dir, config.years)
    graph_dataset, x_mean, x_std = elementwise_scale_node_feature(graph_dataset)
    e_mean, e_std = normalize_edge_features(graph_dataset)

    np.random.shuffle(graph_dataset)

    split_idx = int(config.train_split * len(graph_dataset))
    training_data = graph_dataset[:split_idx]
    test_data = graph_dataset[split_idx:]

    train_dataset = GraphListDataset(training_data)
    loader = DisjointLoader(
        train_dataset,
        batch_size=config.batch_size,
        epochs=config.epochs,
        shuffle=True,
    )

    il_agent = ILAgent()
    loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=False)
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

    avg_train_losses = train_model(il_agent, loader, loss_fn, optimizer)

    if config.show_plots:
        plt.figure(figsize=(10, 6))
        plt.plot(avg_train_losses, marker="o", linestyle="-", color="#1f77b4", label="Train Loss")
        plt.title("Training Loss Curve", fontsize=16, fontweight="bold")
        plt.xlabel("Epoch", fontsize=13, fontweight="bold")
        plt.ylabel("Average Loss", fontsize=13, fontweight="bold")
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.xticks(fontsize=11)
        plt.yticks(fontsize=11)
        plt.legend(fontsize=12)
        plt.tight_layout()
        plt.show()

    y_preds, y_trues = evaluate_model(il_agent, test_data)

    if config.show_plots:
        plt.figure(figsize=(8, 4))
        plt.hist(y_preds.flatten(), bins=30, color="skyblue", edgecolor="k")
        plt.axvline(config.threshold_low, color="red", linestyle="--", label="Low threshold")
        plt.axvline(config.threshold_high, color="green", linestyle="--", label="High threshold")
        plt.title("Prediction Confidence Distribution")
        plt.xlabel("Predicted Value")
        plt.ylabel("Frequency")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    confident_mask = (y_preds >= config.threshold_high) | (y_preds <= config.threshold_low)
    binary_preds = np.where(y_preds >= config.threshold_high, 1, np.where(y_preds <= config.threshold_low, 0, -1))
    correct = binary_preds == y_trues
    correct_confident = correct[confident_mask]
    accuracy = correct_confident.sum() / max(1, confident_mask.sum())
    confident_errors = ((binary_preds != y_trues) & confident_mask).sum()
    uncertain_mask = binary_preds == -1
    uncertainty_rate = uncertain_mask.sum() / np.prod(y_trues.shape)

    print("\nPer-variable confident accuracy:")
    for idx in range(y_trues.shape[1]):
        mask_i = confident_mask[:, idx]
        correct_i = (binary_preds[:, idx] == y_trues[:, idx])[mask_i]
        acc_i = correct_i.mean() if mask_i.sum() > 0 else np.nan
        print(f"  y_{idx + 1}: {acc_i:.4f} accuracy on {mask_i.sum()} confident predictions")

    print("\n=== Confidence-Based Evaluation Summary ===")
    print(f"Thresholded Accuracy (confident preds only): {accuracy:.4f}")
    print(
        "Confidence coverage: "
        f"{confident_mask.sum()} / {np.prod(y_trues.shape)} "
        f"({(confident_mask.sum() / np.prod(y_trues.shape)) * 100:.2f}%)"
    )
    print(f"Confident misclassifications: {confident_errors}")
    print(f"Uncertain predictions: {uncertain_mask.sum()} ({uncertainty_rate * 100:.2f}%)")

    if config.show_plots:
        correct_confident_count = correct_confident.sum()
        confident_error_count = int((binary_preds != y_trues)[confident_mask].sum())
        uncertain_count = uncertain_mask.sum()
        labels = ["Correct (Confident)", "Incorrect (Confident)", "Uncertain"]
        sizes = [correct_confident_count, confident_error_count, uncertain_count]
        colors = ["#4F81BD", "#6BAED6", "#A6CEE3"]

        plt.figure(figsize=(14, 6))
        bars = plt.barh(labels, sizes, color=colors, height=0.4)
        plt.xlabel("Number of Predictions", fontsize=13, fontweight="bold")
        plt.title("Variable-wise Prediction Outcomes", fontsize=14, fontweight="bold")
        for bar, size in zip(bars, sizes):
            plt.text(
                bar.get_width() + 10,
                bar.get_y() + bar.get_height() / 2,
                f"{size}",
                va="center",
                fontsize=12,
                fontweight="bold",
                color="black",
            )
        plt.grid(axis="x", linestyle="--", alpha=0.6)
        plt.xticks(fontsize=11)
        plt.yticks(fontsize=12, fontweight="bold")
        plt.tight_layout()
        plt.show()

    il_agent.save(config.output_dir / "saved_agents" / "il_agent")
    np.save(config.output_dir / "x_mean.npy", x_mean)
    np.save(config.output_dir / "x_std.npy", x_std)
    np.save(config.output_dir / "e_mean.npy", e_mean)
    np.save(config.output_dir / "e_std.npy", e_std)


def parse_years(value: str) -> list[int]:
    if isinstance(value, list):
        return [int(item) for item in value]
    if value.strip().lower() == "all":
        return list(range(2009, 2024))
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def _load_config(config_path: Optional[Path]) -> dict:
    if config_path is None:
        return {}
    with config_path.open("r", encoding="utf-8") as file:
        payload = json.load(file)
    if not isinstance(payload, dict):
        raise ValueError("Config file must contain a JSON object.")
    validate_cs2_train_il_config(payload)
    return payload


def parse_args(argv=None) -> TrainConfig:
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", type=Path, default=None)
    pre_args, remaining = pre_parser.parse_known_args(argv)
    config_values = _load_config(pre_args.config)

    parser = argparse.ArgumentParser(description="Train IL agent for Case Study 2.")
    parser.add_argument("--config", type=Path, default=pre_args.config)
    parser.add_argument("--dataset-dir", type=Path, default=Path(config_values.get("dataset_dir", "./gbd_datasets")))
    parser.add_argument("--output-dir", type=Path, default=Path(config_values.get("output_dir", "./results")))
    years_default = config_values.get("years", "all")
    if isinstance(years_default, list):
        years_default = ",".join(str(year) for year in years_default)
    parser.add_argument("--years", type=str, default=years_default)
    parser.add_argument("--batch-size", type=int, default=int(config_values.get("batch_size", 128)))
    parser.add_argument("--epochs", type=int, default=int(config_values.get("epochs", 100)))
    parser.add_argument("--train-split", type=float, default=float(config_values.get("train_split", 0.85)))
    parser.add_argument("--threshold-high", type=float, default=float(config_values.get("threshold_high", 0.8)))
    parser.add_argument("--threshold-low", type=float, default=float(config_values.get("threshold_low", 0.2)))
    parser.add_argument("--seed", type=int, default=int(config_values.get("seed", 42)))
    default_no_plots = bool(config_values.get("no_plots", False))
    parser.add_argument("--no-plots", action="store_true", default=default_no_plots)

    args = parser.parse_args(remaining)
    return TrainConfig(
        dataset_dir=args.dataset_dir,
        output_dir=args.output_dir,
        years=parse_years(args.years),
        batch_size=args.batch_size,
        epochs=args.epochs,
        train_split=args.train_split,
        threshold_high=args.threshold_high,
        threshold_low=args.threshold_low,
        seed=args.seed,
        show_plots=not args.no_plots,
    )


def main():
    config = parse_args()
    run_training(config)


if __name__ == "__main__":
    main()
