import time
import pandas as pd
from helper_functions.generate_problem_instances import GenerateDataset
import numpy as np
from helper_functions.gbd_vanilla import solve_problem_with_gbd
from helper_functions.gbd_proposed_approach import solve_problem_with_gbd_proposed
from matplotlib import pyplot as plt 
import seaborn as sns

generated_parameters = np.load("./test_parameters/generated_parameters_test.npy", allow_pickle=True)
optimal_solutions_true = np.load("./test_parameters/optimal_solutions_true_test.npy", allow_pickle=True)
optimal_cost_true = np.load("./test_parameters/optimal_cost_true_test.npy", allow_pickle=True)

def evaluate_cost(solution, parameters):
    y = solution[:5]
    x = solution[5:]
    coeff_y1, coeff_y2, coeff_y3, coeff_y4, coeff_y5, _, _, _, _, _ = parameters
    cost = (coeff_y1 * y[0] + coeff_y2 * y[1] + coeff_y3 * y[2] +
            coeff_y4 * y[3] + coeff_y5 * y[4] - 10*x[0] - 15*x[1] -
            15*x[2] + 15*x[3] + 5*x[4] - 20*x[5] +
            np.exp(x[0]) + np.exp(x[1]/1.2) -
            60*np.log(x[3] + x[4] + 1) + 140)
    return cost

initial_guess = [1,0,0,0,0]
eps = 1e-6

results_gbd = []
time_start_gbd = time.time()
for i in range(len(generated_parameters)):
    print(f"[GBD] Solving instance {i+1}")
    time_gbd_cur_init = time.time()
    x, y, lbds_cur, lbds, ubds,  time_mp = solve_problem_with_gbd(
        initial_guess, generated_parameters[i]
    )
    len(lbds) == len(ubds)
    sol = []
    sol.extend(y)
    sol.extend(x)
    time_gbd_cur_end = time.time()
    iterations = len(lbds)
    true_sol = optimal_solutions_true[i]
    cost_gbd = evaluate_cost(sol, generated_parameters[i])
    results_gbd.append({
        "instance_id": i,
        "solution": sol,
        "lbds": lbds,
        "ubds": ubds,
        "time_mp": time_mp,
        "iterations": iterations,
        "time_taken": time_gbd_cur_end - time_gbd_cur_init,
        "absolute_gap": abs(cost_gbd - optimal_cost_true[i]),
        "relative_gap": abs(cost_gbd - optimal_cost_true[i])/(abs(optimal_cost_true[i]) + eps),
        "error_to_true": np.linalg.norm(np.array(sol) - np.array(true_sol))
    })
time_end_gbd = time.time()
total_time_gbd = time_end_gbd - time_start_gbd
avg_time_gbd = total_time_gbd / len(generated_parameters)

print(f"\n[GBD] Total time: {total_time_gbd:.2f}s")
print(f"[GBD] Avg time per instance: {avg_time_gbd:.4f}s")

results_il_rl = []
time_start_il_rl = time.time()
for i in range(len(generated_parameters)):
    print(f"[IL + RL] Solving instance {i+1}")
    time_il_rl_cur_init = time.time()
    x, y, binary_preds, preds, lbd, lbds_cur, ubds_cur, time_mp = solve_problem_with_gbd_proposed(
        initial_guess, generated_parameters[i], 'rl_il_init'
    )
    time_il_rl_cur_end = time.time()
    len(lbds_cur) == len(ubds_cur)
    sol = []
    sol.extend(y)
    sol.extend(x)
    iterations = len(lbds_cur)
    true_sol = optimal_solutions_true[i]
    cost_il_rl = evaluate_cost(sol, generated_parameters[i])
    results_il_rl.append({
        "instance_id": i,
        "solution": sol,
        "predicted_binary": binary_preds,
        "lbds": lbds_cur,
        "ubds": ubds_cur,
        "time_mp": time_mp,
        "lbd": lbd,
        "iterations": iterations,
        "absolute_gap": abs(cost_il_rl - optimal_cost_true[i]),
        "relative_gap": abs(cost_il_rl - optimal_cost_true[i])/(abs(optimal_cost_true[i]) + eps),
        "error_to_true": np.linalg.norm(np.array(sol) - np.array(true_sol)),
        "time_taken": time_il_rl_cur_end - time_il_rl_cur_init
    })
time_end_il_rl = time.time()
total_time_il_rl = time_end_il_rl - time_start_il_rl
avg_time_il_rl = total_time_il_rl / len(generated_parameters)

print(f"\n[IL + RL] Total time: {total_time_il_rl:.2f}s")
print(f"[IL + RL] Avg time per instance: {avg_time_il_rl:.4f}s")


results_il = []
time_start_il = time.time()
for i in range(len(generated_parameters)):
    print(f"[IL] Solving instance {i+1}")
    time_il_cur_init = time.time()
    x, y, binary_preds, preds, lbd, lbds_cur, ubds_cur, time_mp = solve_problem_with_gbd_proposed(
        initial_guess, generated_parameters[i], 'imitation_learning_agent'
    )
    time_il_cur_end = time.time()
    len(lbds_cur) == len(ubds_cur)
    sol = []
    sol.extend(y)
    sol.extend(x)
    iterations = len(lbds_cur)
    true_sol = optimal_solutions_true[i]
    cost_il = evaluate_cost(sol, generated_parameters[i])
    results_il.append({
        "instance_id": i,
        "solution": sol,
        "predicted_binary": binary_preds,
        "lbds": lbds_cur,
        "ubds": ubds_cur,
        "time_mp": time_mp,
        "lbd": lbd,
        "iterations": iterations,
        "absolute_gap": abs(cost_il - optimal_cost_true[i]),
        "relative_gap": abs(cost_il - optimal_cost_true[i])/(abs(optimal_cost_true[i]) + eps),
        "error_to_true": np.linalg.norm(np.array(sol) - np.array(true_sol)),
        "time_taken": time_il_cur_end - time_il_cur_init
    })
time_end_il = time.time()
total_time_il = time_end_il - time_start_il
avg_time_il = total_time_il / len(generated_parameters)

print(f"\n[IL] Total time: {total_time_il:.2f}s")
print(f"[IL] Avg time per instance: {avg_time_il:.4f}s")

results_rl = []
time_start_rl = time.time()
for i in range(len(generated_parameters)):
    print(f"[RL] Solving instance {i+1}")
    time_rl_cur_init = time.time()
    x, y, binary_preds, preds, lbd, lbds_cur, ubds_cur, time_mp = solve_problem_with_gbd_proposed(
        initial_guess, generated_parameters[i], 'rl_random_init'
    )
    time_rl_cur_end = time.time()
    len(lbds_cur) == len(ubds_cur)
    sol = []
    sol.extend(y)
    sol.extend(x)
    iterations = len(lbds_cur)
    true_sol = optimal_solutions_true[i]
    cost_rl = evaluate_cost(sol, generated_parameters[i])
    results_rl.append({
        "instance_id": i,
        "solution": sol,
        "predicted_binary": binary_preds,
        "lbds": lbds_cur,
        "ubds": ubds_cur,
        "time_mp": time_mp,
        "lbd": lbd,
        "iterations": iterations,
        "absolute_gap": abs(cost_rl - optimal_cost_true[i]),
        "relative_gap": abs(cost_rl - optimal_cost_true[i])/(abs(optimal_cost_true[i]) + eps),
        "error_to_true": np.linalg.norm(np.array(sol) - np.array(true_sol)),
        "time_taken": time_rl_cur_end - time_rl_cur_init
    })
time_end_rl = time.time()
total_time_rl = time_end_rl - time_start_rl
avg_time_rl = total_time_rl / len(generated_parameters)

print(f"\n[RL] Total time: {total_time_rl:.2f}s")
print(f"[RL] Avg time per instance: {avg_time_rl:.4f}s")

plot_data = []
for label, results in [
    ("Classical GBD", results_gbd),
    ("Imitation Learning", results_il),
    ("Proposed", results_il_rl),
    ("RL (Random Init)", results_rl),
]:
    for r in results:
        plot_data.append({
            "Method": label,
            "Time (s)": r["time_taken"],
            "MP Time (s)": r["time_mp"],          
            "Iterations": r["iterations"],
            "Absolute Cost Gap": r["absolute_gap"],
            "Relative Cost Gap": r["relative_gap"],
            "Error to True": r["error_to_true"]
        })

df_plot = pd.DataFrame(plot_data)


# ========= Summary stats: mean ± std for total and MP runtimes (and iterations) =========
def mean_std_fmt(series, decimals=2):
    m = series.mean()
    s = series.std(ddof=1)  # sample std
    return f"{m:.{decimals}f} ± {s:.{decimals}f}"

summary_tbl = (
    df_plot
    .groupby("Method")
    .agg(
        total_time_mean=("Time (s)", "mean"),
        total_time_std=("Time (s)", "std"),
        mp_time_mean=("MP Time (s)", "mean"),
        mp_time_std=("MP Time (s)", "std"),
        iters_mean=("Iterations", "mean"),
        iters_std=("Iterations", "std")
    )
    .reset_index()
)

# Pretty columns with "mean ± std"
summary_tbl["Total runtime (s)"] = summary_tbl.apply(
    lambda r: f"{r['total_time_mean']:.2f} ± {r['total_time_std']:.2f}", axis=1
)
summary_tbl["MP runtime (s)"] = summary_tbl.apply(
    lambda r: f"{r['mp_time_mean']:.2f} ± {r['mp_time_std']:.2f}", axis=1
)
summary_tbl["Iterations"] = summary_tbl.apply(
    lambda r: f"{r['iters_mean']:.2f} ± {r['iters_std']:.2f}", axis=1
)

# Keep only the pretty columns for reporting
report_cols = ["Method", "Total runtime (s)", "MP runtime (s)", "Iterations"]
summary_pretty = summary_tbl[report_cols].copy()

print("\n=== Mean ± Std Summary (per instance) ===")
print(summary_pretty.to_string(index=False))

# ========= Convergence summary table =========

# Define tolerance for convergence to "true" solution
TOL_REL = 1e-6  # adjust if needed

# Group by Method
conv_summary = (
    df_plot.groupby("Method")
    .agg(
        num_converged=("Relative Cost Gap", lambda g: np.sum(g <= TOL_REL)),
        total_instances=("Relative Cost Gap", "size"),
        mean_rel_gap=("Relative Cost Gap", "mean"),
        std_rel_gap=("Relative Cost Gap", "std")
    )
    .reset_index()
)

# Compute percentage converged
conv_summary["Convergence Rate (%)"] = (
    conv_summary["num_converged"] / conv_summary["total_instances"] * 100
).round(1)

# Round the gap stats for neatness
conv_summary["Mean Rel. Gap"] = conv_summary["mean_rel_gap"].round(6)
conv_summary["Std Rel. Gap"] = conv_summary["std_rel_gap"].round(6)

# Keep only needed columns
conv_table = conv_summary[[
    "Method",
    "num_converged",
    "total_instances",
    "Convergence Rate (%)",
    "Mean Rel. Gap",
    "Std Rel. Gap"
]]

print("\n=== Convergence Summary ===")
print(conv_table.to_string(index=False))





# Group stats
rel_gap_stats = df_plot.groupby("Method")["Relative Cost Gap"].agg(["mean", "std"]).reset_index()
rel_gap_stats.columns = ["Method", "Mean Relative Gap", "Std Dev"]

# Extract values
methods = rel_gap_stats["Method"]
means = rel_gap_stats["Mean Relative Gap"]
stds = rel_gap_stats["Std Dev"]

# Construct asymmetric error bars: 0 below, std above
lower_errors = np.zeros_like(stds)
upper_errors = stds.to_numpy()
asymmetric_error = [lower_errors, upper_errors]

# # Plot
# plt.figure(figsize=(9, 6))
# plt.grid(False)
# bars = plt.bar(
#     methods,
#     means,
#     yerr=asymmetric_error,
#     capsize=5,
#     color=sns.color_palette("Set2", len(means))
# )

# # Annotate top of error bars with mean (σ = std)
# for bar, mean, std in zip(bars, means, stds):
#     height = bar.get_height()
#     top = height + std
#     plt.text(
#         bar.get_x() + bar.get_width() / 2,
#         top + 0.0002,
#         f"{mean:.3f} (σ={std:.3f})",
#         ha='center',
#         va='bottom',
#         fontsize=10,
#         fontweight='bold'
#     )

# plt.title("Mean Relative Cost Gap with Standard Deviation", fontsize=13, fontweight="bold")
# plt.ylabel("Mean Relative Gap", fontsize=11)
# plt.xlabel("")
# plt.axhline(0, color='black', linewidth=1)
# plt.ylim(0, (means + stds).max() * 1.15)
# plt.tight_layout()
# # plt.savefig("comparative_study_ex_2_relative_gap.png", dpi=300)
# plt.show()

# Example dictionary of total times in seconds
total_times = {
    "Classical GBD": total_time_gbd,
    "Imitation Learning": total_time_il,
    "Proposed": total_time_il_rl,
    "RL (Random Init)": total_time_rl
}

# Convert to lists for plotting
methods = list(total_times.keys())
times = list(total_times.values())

# # Plot
# plt.figure(figsize=(10, 8))
# plt.grid(False)
# bars = plt.bar(methods, times, color=sns.color_palette("Set2", len(methods)))

# # Annotate each bar with the time (formatted nicely)
# for bar, time in zip(bars, times):
#     height = bar.get_height()
#     plt.text(
#         bar.get_x() + bar.get_width()/2,
#         height + 1,  # adjust spacing above the bar
#         f"{time:.2f} s",
#         ha='center',
#         va='bottom',
#         fontsize=10,
#         fontweight='bold'
#     )

# # Title and labels
# plt.title("Total Runtime per Method (100 Instances)", fontsize=13, fontweight="bold")
# plt.ylabel("Total Time (seconds)", fontsize=11)
# plt.xlabel("")
# # plt.xticks(rotation=10)
# plt.ylim(0, max(times) * 1.15)
# plt.tight_layout()
# # plt.savefig("comparative_study_ex_2_total_runtime.png", dpi=300)
# plt.show()


# --- Improvement in Total Runtime Over Classical GBD ---
def compute_improvement(baseline, other):
    return 100 * (baseline - other) / baseline

total_improvements = {
    "IL": compute_improvement(total_time_gbd, total_time_il),
    "RL (IL init)": compute_improvement(total_time_gbd, total_time_il_rl),
    "RL (Random Init)": compute_improvement(total_time_gbd, total_time_rl)
}

print("\n[Total Runtime Improvement Over Classical GBD]")
for method, imp in total_improvements.items():
    print(f"{method}: {imp:.2f}% faster than Classical GBD")


# --- Median-Based Runtime per Method (Matplotlib with bar width) ---
median_times = df_plot.groupby("Method")["Time (s)"].median().reset_index()


# --- Improvement in Median Runtime Over Classical GBD ---
baseline_median = median_times[median_times["Method"] == "Classical GBD"]["Time (s)"].values[0]
median_improvements = {}
for idx, row in median_times.iterrows():
    if row["Method"] != "Classical GBD":
        imp = compute_improvement(baseline_median, row["Time (s)"])
        median_improvements[row["Method"]] = imp

print("\n[Median Runtime Improvement Over Classical GBD (Per Instance)]")
for method, imp in median_improvements.items():
    print(f"{method}: {imp:.2f}% faster (median-based)")

# # Define method names and their corresponding results
# methods_results = {
#     "Imitation Learning": results_il,
#     "Proposed": results_il_rl,
#     "RL (Random Init)": results_rl
# }

# # Set numerical tolerance for considering a decrease
# TOL = 1e-5

# # Store violation stats and violating rows
# all_violation_stats = {}
# all_violators = {}

# # Function to check monotonicity and return violating rows
# def monotonic_violation_stats(results, method_name):
#     violation_counts = []
#     violating_rows = []
#     for r in results:
#         lbds = r.get("lbds", [])
#         if len(lbds) < 2:
#             continue
#         violations = sum((lbds[i+1] + TOL) < lbds[i] for i in range(len(lbds)-1))
#         violation_counts.append(violations)
#         if violations > 0:
#             violating_rows.append({
#                 "instance_id": r["instance_id"],
#                 "violations": violations,
#                 "lbds": lbds,
#                 "predicted_binary": r.get("predicted_binary", []),
#                 "method": method_name
#             })

#     print(f"[{method_name}] Monotonic Violation Summary:")
#     print(f"  Instances with any violation: {sum(v > 0 for v in violation_counts)} / {len(violation_counts)}")
#     print(f"  Avg violations per instance: {np.mean(violation_counts):.2f}\n")
#     return violation_counts, violating_rows

# # Run for all methods
# for method, results in methods_results.items():
#     counts, violators = monotonic_violation_stats(results, method)
#     all_violation_stats[method] = counts
#     all_violators[method] = violators

# # Combine violating rows into a single DataFrame
# violating_df = pd.DataFrame([
#     {
#         "Method": v["method"],
#         "Instance ID": v["instance_id"],
#         "Num Violations": v["violations"],
#         "LBDs": v["lbds"],
#         "Binary Assignment": v["predicted_binary"]
#     }
#     for method_violators in all_violators.values()
#     for v in method_violators
# ])



