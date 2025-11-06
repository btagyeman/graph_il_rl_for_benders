import time
import os
import pandas as pd
import numpy as np
from helper_functions.gbd_vanilla import solve_problem_with_gbd
from helper_functions.gbd_proposed_approach import solve_problem_with_gbd_proposed
from matplotlib import pyplot as plt

# ---------------- EPS/PDF output settings ----------------
EXPORT_EPS = True
plt.rcParams.update({
    "font.family": "Times New Roman",
    "text.usetex": False,      # safer for EPS
    "ps.fonttype": 42,         # embed TrueType as Type42
    "pdf.fonttype": 42,
})

generated_parameters = np.load("./test_parameters/generated_parameters_test.npy", allow_pickle=True)
optimal_solutions_true = np.load("./test_parameters/optimal_solutions_true_test.npy", allow_pickle=True)
optimal_cost_true = np.load("./test_parameters/optimal_cost_true_test.npy", allow_pickle=True)

def evaluate_cost(solution, parameters):
    y = solution[:5]
    x = solution[5:]
    coeff_y1, coeff_y2, coeff_y3, coeff_y4, coeff_y5, _, _, _, _, _ = parameters
    cost = (coeff_y1 * y[0] + coeff_y2 * y[1] + coeff_y3 * y[2]
            + coeff_y4 * y[3] + coeff_y5 * y[4]
            - 10*x[0] - 15*x[1] - 15*x[2] + 15*x[3] + 5*x[4] - 20*x[5]
            + np.exp(x[0]) + np.exp(x[1]/1.2) - 60*np.log(x[3] + x[4] + 1) + 140)
    return cost

initial_guess = [1, 0, 0, 0, 0]
eps = 1e-6

# ---------------- Solve all instances ----------------
results_gbd = []
time_start_gbd = time.time()
for i in range(len(generated_parameters)):
    print(f"[GBD] Solving instance {i+1}")
    t0 = time.time()
    x, y, lbds_cur, lbds, ubds, time_mp = solve_problem_with_gbd(initial_guess, generated_parameters[i])
    sol = [*y, *x]
    t1 = time.time()
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
        "time_taken": t1 - t0,
        "absolute_gap": abs(cost_gbd - optimal_cost_true[i]),
        "relative_gap": abs(cost_gbd - optimal_cost_true[i])/(abs(optimal_cost_true[i]) + eps),
        "error_to_true": np.linalg.norm(np.array(sol) - np.array(true_sol))
    })
avg_time_gbd = (time.time() - time_start_gbd) / len(generated_parameters)

results_il_rl = []
time_start_il_rl = time.time()
for i in range(len(generated_parameters)):
    print(f"[IL + RL] Solving instance {i+1}")
    t0 = time.time()
    x, y, binary_preds, preds, lbd, lbds_cur, ubds_cur, time_mp = solve_problem_with_gbd_proposed(
        initial_guess, generated_parameters[i], 'rl_il_init'
    )
    t1 = time.time()
    sol = [*y, *x]
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
        "time_taken": t1 - t0
    })
avg_time_il_rl = (time.time() - time_start_il_rl) / len(generated_parameters)

results_il = []
time_start_il = time.time()
for i in range(len(generated_parameters)):
    print(f"[IL] Solving instance {i+1}")
    t0 = time.time()
    x, y, binary_preds, preds, lbd, lbds_cur, ubds_cur, time_mp = solve_problem_with_gbd_proposed(
        initial_guess, generated_parameters[i], 'imitation_learning_agent'
    )
    t1 = time.time()
    sol = [*y, *x]
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
        "time_taken": t1 - t0
    })
avg_time_il = (time.time() - time_start_il) / len(generated_parameters)

results_rl = []
time_start_rl = time.time()
for i in range(len(generated_parameters)):
    print(f"[RL] Solving instance {i+1}")
    t0 = time.time()
    x, y, binary_preds, preds, lbd, lbds_cur, ubds_cur, time_mp = solve_problem_with_gbd_proposed(
        initial_guess, generated_parameters[i], 'rl_random_init'
    )
    t1 = time.time()
    sol = [*y, *x]
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
        "time_taken": t1 - t0
    })
avg_time_rl = (time.time() - time_start_rl) / len(generated_parameters)

# ---------------- Build plot data ----------------
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

def count_confident_assignments(results, method_label):
    confident = []
    free = []
    for r in results:
        bin_pred = np.array(r["predicted_binary"])
        confident.append(np.sum(bin_pred != -1))
        free.append(np.sum(bin_pred == -1))
    total_conf = int(np.sum(confident))
    total_free = int(np.sum(free))
    return {"Method": method_label, "Confident": total_conf, "Free": total_free, "Total": total_conf + total_free}

count_il    = count_confident_assignments(results_il, "Imitation Learning")
count_il_rl = count_confident_assignments(results_il_rl, "Proposed")
count_rl    = count_confident_assignments(results_rl, "RL (Random Init)")

assignment_df = pd.DataFrame([count_il, count_il_rl, count_rl])
melted_df = assignment_df.melt(
    id_vars=["Method", "Total"],
    value_vars=["Confident", "Free"],
    var_name="Assignment Type",
    value_name="Count"
)
melted_df["Percentage"] = (melted_df["Count"] / melted_df["Total"]) * 100

# ---------------- Save processed CSV + LaTeX rows ----------------
os.makedirs("data/processed", exist_ok=True)
methods = ["Imitation Learning", "Proposed", "RL (Random Init)"]
pivot_df = (melted_df
            .pivot(index="Method", columns="Assignment Type", values="Percentage")
            .reindex(methods))
out_df = (assignment_df.set_index("Method")
          .loc[methods, ["Confident", "Free", "Total"]]
          .join(pivot_df.rename(columns={"Confident": "Confident_pct", "Free": "Free_pct"}))
          .reset_index())
out_df = out_df[["Method", "Confident", "Free", "Total", "Confident_pct", "Free_pct"]]
out_df["Confident_pct"] = out_df["Confident_pct"].round(1)
out_df["Free_pct"] = out_df["Free_pct"].round(1)
csv_path = "data/processed/figB_assignment_frequencies.csv"
out_df.to_csv(csv_path, index=False)
print(f"[saved] {csv_path}")

print("\n% ===== LaTeX for SI (Figure B) =====")
print("\\begin{table}[h]")
print("\\centering")
print("\\caption{Percentage of confident vs. free assignments per agent (Figure B).}")
print("\\small")
print("\\begin{tabular}{l r r r r r}")
print("\\toprule")
print("\\textbf{Agent} & \\textbf{Confident} & \\textbf{Free} & \\textbf{Total} & \\textbf{Confident (\\%)} & \\textbf{Free (\\%)}\\\\")
print("\\midrule")
for _, r in out_df.iterrows():
    print(f"{r['Method']} & {int(r['Confident'])} & {int(r['Free'])} & {int(r['Total'])} & {r['Confident_pct']:.1f} & {r['Free_pct']:.1f}\\\\")
print("\\bottomrule")
print("\\end{tabular}")
print("\\end{table}")

# ---------------- Plot (PDF + EPS) ----------------
colors = {"Confident": "#4C72B0", "Free": "#DD8452"}
conf = pivot_df["Confident"].values
free = pivot_df["Free"].values

x = np.arange(len(methods))
bar_width = 0.32

fig, ax = plt.subplots(figsize=(7.5, 5.5))
bars_conf = ax.bar(x - bar_width/2, conf, width=bar_width, label="Confident", color=colors["Confident"])
bars_free = ax.bar(x + bar_width/2, free, width=bar_width, label="Free", color=colors["Free"])

def add_labels(bars):
    for b in bars:
        h = b.get_height()
        ax.text(b.get_x() + b.get_width()/2, h + 1, f"{h:.1f}%",
                ha="center", va="bottom", fontsize=11, fontweight="bold")

add_labels(bars_conf)
add_labels(bars_free)

ax.set_title("Percentage of confident vs free assignments per agent", fontsize=12, fontweight="bold")
ax.set_ylabel("Percentage of total assignments (%)", fontsize=12, fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels(methods, fontsize=11, fontweight="bold")
ax.set_ylim(0, 105)
legend = ax.legend(title="Assignment type", fontsize=10, title_fontsize=11, frameon=False, ncol=2, loc="upper right")
legend.get_title().set_fontweight('bold')
for label in ax.get_yticklabels() + ax.get_xticklabels():
    label.set_fontweight("bold")
for text in legend.get_texts():
    text.set_fontweight('bold')

ax.grid(False)
for spine in ax.spines.values():
    spine.set_linewidth(2)
    spine.set_color("black")

plt.tight_layout()
os.makedirs("final_plots", exist_ok=True)
plt.savefig("final_plots/assignment_distribution_grouped_times.pdf", dpi=300, bbox_inches="tight")
if EXPORT_EPS:
    plt.savefig("final_plots/assignment_distribution_grouped_times.eps", format="eps", bbox_inches="tight")
print("[saved] final_plots/assignment_distribution_grouped_times.pdf")
if EXPORT_EPS:
    print("[saved] final_plots/assignment_distribution_grouped_times.eps")
plt.show()
