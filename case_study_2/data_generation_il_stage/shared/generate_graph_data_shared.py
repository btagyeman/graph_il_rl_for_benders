from __future__ import annotations

import argparse
import importlib
import os
import pickle
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional


@dataclass
class GenerationConfig:
    year: int
    year_dir: Path
    output_path: Optional[Path] = None
    plot_trajectory: bool = True


@contextmanager
def working_directory(path: Path):
    current = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(current)


def _clear_cached_year_modules() -> None:
    prefixes = ("common", "helper_functions")
    for name in list(sys.modules):
        if name in prefixes or name.startswith("common.") or name.startswith("helper_functions."):
            del sys.modules[name]


def _prepare_imports(year_dir: Path):
    year_dir_str = str(year_dir)
    if year_dir_str not in sys.path:
        sys.path.insert(0, year_dir_str)

    _clear_cached_year_modules()

    np = importlib.import_module("numpy")
    plt = importlib.import_module("matplotlib.pyplot")
    spektral_data = importlib.import_module("spektral.data")

    temporal = importlib.import_module("common.actual_field.temporal_spatial_parameters_field")
    graph_representation = importlib.import_module("helper_functions.graph_representation")
    richards_eq = importlib.import_module("common.actual_field.richards_equation_field")
    season_utils = importlib.import_module("common.utils_season_long")
    season_long_sim = importlib.import_module("helper_functions.season_long_simulation")

    return {
        "np": np,
        "plt": plt,
        "Dataset": spektral_data.Dataset,
        "spatialVariables_1D_act": temporal.spatialVariables_1D_act,
        "pars_mz1": temporal.pars_mz1,
        "create_graph_representation": graph_representation.create_graph_representation,
        "vwc_fun": richards_eq.volMoistureAllNodes,
        "compute_root_zone_moisture": season_utils.compute_root_zone_moisture,
        "obtain_soil_moisture": season_utils.obtain_soil_moisture,
        "run_irrigation_scheduler": season_long_sim.run_irrigation_scheduler,
    }


def run_year_generation(config: GenerationConfig) -> Path:
    year_dir = config.year_dir.resolve()
    with working_directory(year_dir):
        imports = _prepare_imports(year_dir)

        np = imports["np"]
        plt = imports["plt"]
        Dataset = imports["Dataset"]
        spatial_variables = imports["spatialVariables_1D_act"]
        pars_mz1 = imports["pars_mz1"]
        create_graph = imports["create_graph_representation"]
        vwc_fun = imports["vwc_fun"]
        compute_root_zone_moisture = imports["compute_root_zone_moisture"]
        obtain_soil_moisture = imports["obtain_soil_moisture"]
        run_irrigation_scheduler = imports["run_irrigation_scheduler"]

        os.makedirs("./gbd_datasets/", exist_ok=True)

        _, _, total_nodes = spatial_variables()
        soil_pars_mz1 = pars_mz1()

        data_min_mz1 = np.loadtxt("../../trained_lstm_models/model_weights/data_min_mz1_vrd_lai.txt")
        data_max_mz1 = np.loadtxt("../../trained_lstm_models/model_weights/data_max_mz1_vrd_lai.txt")
        rain_act = np.loadtxt("./common/weather_data/dRAIN.txt")

        initial_state_mz1 = np.loadtxt("./common/initial_states/Init_state_rz_mz1_vrd_lai.txt")
        initial_irrigation_mz1 = np.loadtxt("./common/initial_states/Init_irrig_mz1_vrd_lai.txt")
        initial_irrigation_mz1 = initial_irrigation_mz1 * 86400
        all_head_values_mz1 = np.loadtxt("./common/initial_states/Init_state_all_mz1_vrd_lai.txt")

        class GenerateDataset(Dataset):
            def __init__(self, **kwargs):
                self.measurement_covariance = 19.25
                self.process_covariance = 0.05 * np.eye(21)
                self.covariance_mz1 = 15.9 * np.ones((21, 21))
                self.simulation_period = 123
                self.time_instants = np.arange(self.simulation_period)
                self.prediction_horizon = 14
                self.sequence_length = 5
                self.horizon_length_weather = int(self.sequence_length + self.prediction_horizon)
                self.rooting_depths = 0.5 * np.ones(160)
                self.rain_noisy_all = np.zeros((self.simulation_period, self.horizon_length_weather))
                self.ref_evap_noisy_all = np.zeros((self.simulation_period, self.horizon_length_weather))
                self.guesses_x = np.array(
                    [0.224, 0.252, 0.246, 0.242, 0.234, 0.232, 0.229, 0.226, 0.224, 0.221, 0.219, 0.216, 0.213, 0.210]
                )
                self.guesses_u = -0.003887 * np.ones(self.prediction_horizon)
                self.guesses_c = np.ones(self.prediction_horizon)
                self.states_list_mz1 = []
                self.irrigation_list_mz1 = []
                self.set_initial_vwc_and_irrig()
                self.prescribed_irrigation_closed_loop_mz1 = []
                self.state_trajectory_closed_loop_mz1 = []
                self.state_trajectory_open_loop_mz1 = []
                self.irrigation_decisions_closed_loop = []
                self.state_trajectory_closed_loop_mz1.append(initial_state_mz1[-1])
                initial_moisture = compute_root_zone_moisture(
                    obtain_soil_moisture(1.05 * all_head_values_mz1, soil_pars_mz1, vwc_fun),
                    0.50,
                )
                self.state_trajectory_open_loop_mz1.append(initial_moisture)
                self.factor_mz1 = initial_moisture / initial_state_mz1[4]
                self.all_states_mz1 = np.zeros((self.simulation_period + 1, total_nodes))
                self.all_states_mz1_est = np.zeros((self.simulation_period + 1, total_nodes))
                self.all_states_mz1[0, :] = all_head_values_mz1
                self.all_states_mz1_est[0, :] = 1.05 * all_head_values_mz1
                self.graph_features = None
                self.cut_order = None
                self.optimal_values = None
                self.n_variables = 14
                self.lbd = -10e18
                self.ubd = 10e18
                self.lower_zone_mz1 = 0.200
                self.upper_zone_mz1 = 0.280
                self.epsilon = 0.5
                self.r_c = 5
                self.r_u = -50
                super().__init__(**kwargs)

            def set_initial_vwc_and_irrig(self):
                for idx in range(self.sequence_length - 1):
                    self.states_list_mz1 += [initial_state_mz1[idx]]
                    self.irrigation_list_mz1 += [initial_irrigation_mz1[idx]]
                self.states_list_mz1 += [
                    compute_root_zone_moisture(
                        obtain_soil_moisture(1.05 * all_head_values_mz1, soil_pars_mz1, vwc_fun),
                        0.50,
                    )
                ]

            def plot_trajectory(self):
                u_prescribed_mz1 = np.array(self.prescribed_irrigation_closed_loop_mz1) * 1000
                time_stamps = np.arange(1, self.simulation_period + 2)
                time_stamps_input = np.arange(1, self.simulation_period + 1)
                fig, axs = plt.subplots(2, figsize=(10, 10))
                axs[0].bar(time_stamps_input, -u_prescribed_mz1, width=1.2, facecolor=None)
                axs[0].bar(time_stamps_input, rain_act[4 : len(u_prescribed_mz1) + 4], color="red", width=1.0)
                axs[0].legend(["Proposed", "Rain"])
                axs[0].set_title("Irrigation -  MZ 1")
                axs[0].set_ylabel("Irrigation (mm/day)")
                axs[0].set_xlabel("Time (days)")
                axs[0].set_xlim(xmin=1)
                axs[1].plot(time_stamps, self.state_trajectory_closed_loop_mz1, color="blue", linewidth=1.8)
                axs[1].plot(
                    time_stamps,
                    self.upper_zone_mz1 * np.ones(len(self.state_trajectory_open_loop_mz1)),
                    color="darkcyan",
                    linestyle="dashdot",
                    linewidth=2.5,
                )
                axs[1].plot(
                    time_stamps,
                    self.lower_zone_mz1 * np.ones(len(self.state_trajectory_open_loop_mz1)),
                    color="darkmagenta",
                    linestyle="dashdot",
                    linewidth=2.5,
                )
                axs[1].plot(
                    time_stamps,
                    0.12 * np.ones(len(self.state_trajectory_open_loop_mz1)),
                    color="red",
                    linestyle="dashdot",
                    linewidth=2.5,
                )
                axs[1].legend(["state Trajectory", "FC", "Threshold", "PWP"])
                axs[1].set_title("Root Zone Soil Moisture - MZ 1")
                axs[1].set_ylabel("Volumetric moisture")
                axs[1].set_xlabel("Time (days)")
                axs[1].set_xlim(xmin=1)
                fig.tight_layout()
                fig.savefig("Irrigation_trajectory_mz1.png", dpi=300)
                plt.show()

            def generate_graph_data(self):
                return create_graph(self)

            def read(self):
                print("Generating GBD dataset: Starting...")
                entire_dataset = []
                for instant in self.time_instants:
                    run_irrigation_scheduler(self, instant)
                    entire_dataset.extend(self.generate_graph_data())
                print("Generating GBD dataset: Completed\n")
                return entire_dataset

        print(f"Running the season long simulation to generate the dataset for year {config.year}...")
        dataset = GenerateDataset()

        if config.plot_trajectory:
            dataset.plot_trajectory()

        output_path = config.output_path
        if output_path is None:
            output_path = Path(f"./gbd_datasets/graph_gbd_dataset_scheduler_{config.year}.pkl")
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "wb") as output_file:
            pickle.dump(dataset.graphs, output_file)

        print(f"Dataset generated and saved as '{output_path.name}'")
        return output_path.resolve()


def parse_args(argv: Optional[List[str]] = None):
    parser = argparse.ArgumentParser(description="Generate Case Study 2 graph dataset for a specific year.")
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--year-dir", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, default=None)
    parser.add_argument("--no-plot", action="store_true")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    run_year_generation(
        GenerationConfig(
            year=args.year,
            year_dir=args.year_dir,
            output_path=args.output_path,
            plot_trajectory=not args.no_plot,
        )
    )
    return 0


def main_for_year(default_year: int, year_dir: Path, argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=f"Generate Case Study 2 graph dataset for year {default_year}.")
    parser.add_argument("--output-path", type=Path, default=None)
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args(argv)

    run_year_generation(
        GenerationConfig(
            year=default_year,
            year_dir=year_dir,
            output_path=args.output_path,
            plot_trajectory=not args.no_plot,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
