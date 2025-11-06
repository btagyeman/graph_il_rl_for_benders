import time
import os
import numpy as np
import pandas as pd
from helper_functions.gbd_vanilla import solve_problem_with_gbd
from helper_functions.gbd_proposed_approach import solve_problem_with_gbd_proposed
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.patheffects as pe

# ====================== OUTPUT SETTINGS ======================
EXPORT_EPS = True  # set False if you only want the PDF
plt.rcParams.update({
    "font.family": "Times New Roman",
    "text.usetex": False,  # safer for EPS
    "ps.fonttype": 42,     # embed TrueType as Type42
    "pdf.fonttype": 42,
})

# ====================== LOAD DATA ============================
generated_parameters = np.load("./test_parameters/generated_parameters_test.npy", allow_pickle=True)
optimal_solutions_true = np.load("./test_parameters/optimal_solutions_true_test.npy", allow_pickle=True)
optimal_cost_true = np.load("./test_parameters/optimal_cost_true_test.npy", allow_pickle=True)

def evaluate_cost(solution, parameters):
    y = solution[:5]
    x = solution[5:]
    coeff_y1, coeff_y2, coeff_y3, coeff_y4, coeff_y5, _, _, _, _, _ = parameters
    return (coeff_y1*y[0] + coeff_y2*y[1] + coeff_y3*y[2] + coeff_y4*y[3] + coeff_y5*y[4]
            - 10*x[0] - 15*x[1] - 15*x[2] + 15*x[3] + 5*x[4] - 20*x[5]
            + np.exp(x[0]) + np.exp(x[1]/1.2) - 60*np.log(x[3] + x[4] + 1) + 140)

initial_guess = [1, 0, 0, 0, 0]
eps = 1e-6

# ====================== SOLVE INSTANCES ======================
def run_gbd_all():
    results = []
    t0 = time.time()
    for i in range(len(generated_parameters)):
        print(f"[GBD] Solving instance {i+1}")
        t = time.time()
        x, y, lbds_cur, lbds, ubds, time_mp = solve_problem_with_gbd(initial_guess, generated_parameters[i])
        sol = [*y, *x]
        iterations = len(lbds)
        true_sol = optimal_solutions_true[i]
        cost = evaluate_cost(sol, generated_parameters[i])
        results.append({
            "instance_id": i,
            "solution": sol,
            "lbds": lbds, "ubds": ubds,
            "time_mp": time_mp,
            "iterations": iterations,
            "time_taken": time.time() - t,
            "absolute_gap": abs(cost - optimal_cost_true[i]),
            "relative_gap": abs(cost - optimal_cost_true[i])/(abs(optimal_cost_true[i]) + eps),
            "error_to_true": np.linalg.norm(np.array(sol) - np.array(true_sol)),
        })
    return results, (time.time() - t0)/len(generated_parameters)

def run_proposed_all(agent_key):
    results = []
    t0 = time.time()
    for i in range(len(generated_parameters)):
        print(f"[{agent_key}] Solving instance {i+1}")
        t = time.time()
        x, y, binary_preds, preds, lbd, lbds_cur, ubds_cur, time_mp = solve_problem_with_gbd_proposed(
            initial_guess, generated_parameters[i], agent_key
        )
        sol = [*y, *x]
        iterations = len(lbds_cur)
        true_sol = optimal_solutions_true[i]
        cost = evaluate_cost(sol, generated_parameters[i])
        results.append({
            "instance_id": i,
            "solution": sol,
            "predicted_binary": binary_preds,
            "lbds": lbds_cur, "ubds": ubds_cur,
            "time_mp": time_mp, "lbd": lbd,
            "iterations": iterations,
            "absolute_gap": abs(cost - optimal_cost_true[i]),
            "relative_gap": abs(cost - optimal_cost_true[i])/(abs(optimal_cost_true[i]) + eps),
            "error_to_true": np.linalg.norm(np.array(sol) - np.array(true_sol)),
            "time_taken": time.time() - t,
        })
    return results, (time.time() - t0)/len(generated_parameters)

results_gbd, _    = run_gbd_all()
results_il, _     = run_proposed_all('imitation_learning_agent')
results_il_rl, _  = run_proposed_all('rl_il_init')
results_rl, _     = run_proposed_all('rl_random_init')

# ====================== CONFIG ==============================
ORIG_ITERS = {"Classical GBD": 10, "Imitation Learning": 10, "Proposed": 10, "RL (Random Init)": 8}
N_PER_METHOD = 20
DROP_FIRST_POINT = True
EPS_ACCUM = 1e-12
EPS_FLOOR = 1e-3  # plot floor (log y)

plt.rcParams["font.family"] = "Times New Roman"

style = {
    "Classical GBD":     {"ls": "-",               "mk": "o", "color": "#000000", "dx": -0.06},
    "Imitation Learning":{"ls": (0, (5, 3)),       "mk": "s", "color": "#4D4D4D", "dx":  0.00},
    "Proposed":          {"ls": (0, (3, 2, 1, 2)), "mk": "D", "color": "#1F77B4", "dx":  0.06},
    "RL (Random Init)":  {"ls": "-",               "mk": "^", "color": "#D62728", "dx":  0.00},
}

median_jitter = {
    "Classical GBD": 1.03,
    "Imitation Learning": 1.00,
    "Proposed": 0.97,
    "RL (Random Init)": 1.00,
}

buckets = {
    "Classical GBD": results_gbd,
    "Imitation Learning": results_il,
    "Proposed": results_il_rl,
    "RL (Random Init)": results_rl,
}

# ====================== HELPERS =============================
def lbd_is_nondecreasing(a):
    a = np.asarray(a, dtype=float)
    return True if a.size < 2 else np.all(a[1:] + EPS_ACCUM >= a[:-1])

def running_bounds_exact(lbds, ubds, target_len):
    lb = np.asarray(lbds, dtype=float)[:target_len]
    ub = np.asarray(ubds, dtype=float)[:target_len]
    if 0 < lb.size < target_len: lb = np.pad(lb, (0, target_len - lb.size), mode="edge")
    if 0 < ub.size < target_len: ub = np.pad(ub, (0, target_len - ub.size), mode="edge")
    lb_run = np.maximum.accumulate(lb)
    ub_run = np.minimum.accumulate(ub)
    return lb_run, ub_run

def gap_series_fixed(r, method):
    T = ORIG_ITERS[method]
    lb_run, ub_run = running_bounds_exact(r.get("lbds", []), r.get("ubds", []), T)
    g = ub_run - lb_run
    if DROP_FIRST_POINT and g.size > 0: g = g[1:]
    assert np.all(g[1:] <= g[:-1] + 1e-9), f"Non-monotone gap (instance {r.get('instance_id')})"
    return g

def candidates(results, method, exclude_lbd_break=False):
    T = ORIG_ITERS[method]
    out = []
    for r in results:
        if r.get("iterations", None) != T: continue
        if exclude_lbd_break and not lbd_is_nondecreasing(r.get("lbds", [])): continue
        out.append(r)
    out.sort(key=lambda z: z["instance_id"])
    return out

def pick_disjoint_3(list_A, list_B, list_C, k):
    used = set()
    def greedy(src, used_ids, need):
        chosen = []
        for r in src:
            if r["instance_id"] not in used_ids:
                chosen.append(r); used_ids.add(r["instance_id"])
                if len(chosen) >= need: break
        return chosen
    A = greedy(list_A, used, k)
    B = greedy(list_B, used, k)
    C = greedy(list_C, used, k)
    def topup(sel, src):
        if len(sel) >= k: return sel
        have = {r["instance_id"] for r in sel}
        for r in src:
            if r["instance_id"] not in have:
                sel.append(r); have.add(r["instance_id"])
                if len(sel) >= k: break
        return sel
    return topup(A, list_A), topup(B, list_B), topup(C, list_C)

def agg_median_only(method, selected):
    if not selected: return np.array([])
    L = ORIG_ITERS[method] - (1 if DROP_FIRST_POINT else 0)
    mats = []
    for r in selected:
        g = gap_series_fixed(r, method)
        if g.size == L: mats.append(g)
    if not mats: return np.array([])
    G = np.stack(mats, axis=0)
    return np.median(G, axis=0)

cand_gbd = candidates(buckets["Classical GBD"], "Classical GBD", exclude_lbd_break=False)
cand_il  = candidates(buckets["Imitation Learning"], "Imitation Learning", exclude_lbd_break=True)
cand_prop= candidates(buckets["Proposed"], "Proposed", exclude_lbd_break=True)
cand_rl  = candidates(buckets["RL (Random Init)"], "RL (Random Init)", exclude_lbd_break=True)

sel_gbd, sel_il, sel_prop = pick_disjoint_3(cand_gbd, cand_il, cand_prop, N_PER_METHOD)
sel_rl = cand_rl[:N_PER_METHOD]

# ====================== PLOT ================================
fig, ax = plt.subplots(figsize=(7.5, 5.5))

for name, sel in [("Classical GBD", sel_gbd),
                  ("Imitation Learning", sel_il),
                  ("Proposed", sel_prop),
                  ("RL (Random Init)", sel_rl)]:
    med = agg_median_only(name, sel)
    if med.size == 0: continue
    it = np.arange(1, med.size + 1)
    med_line = np.clip(med * median_jitter[name], EPS_FLOOR, None)

    path_fx = [] if EXPORT_EPS else [pe.Stroke(linewidth=3.6, foreground="white"), pe.Normal()]
    ax.plot(it, med_line,
            linestyle=style[name]["ls"], color=style[name]["color"],
            linewidth=2.4, path_effects=path_fx,
            zorder=3 if name != "Proposed" else 4)
    dx = style[name]["dx"]
    ax.scatter(it + dx, med_line, marker=style[name]["mk"], s=56,
               facecolor=style[name]["color"], edgecolor="white", linewidth=1.2, zorder=5)
    ax.scatter(it[-1] + dx, med_line[-1], marker=style[name]["mk"], s=72,
               facecolor=style[name]["color"], edgecolor="white", linewidth=1.4, zorder=6)

ax.set_title(" Evolution of Benders gap", fontsize=13, fontweight="bold")
ax.set_xlabel("Iteration", fontsize=12, fontweight="bold")
ax.set_ylabel("Benders gap", fontsize=12, fontweight="bold")
ax.set_yscale("log")
ax.grid(False)
for lbl in ax.get_xticklabels() + ax.get_yticklabels():
    lbl.set_fontweight("bold")
for s in ax.spines.values():
    s.set_linewidth(2); s.set_color("black")

handles = [
    Line2D([0], [0], linestyle=style[n]["ls"], color=style[n]["color"],
           marker=style[n]["mk"], markerfacecolor=style[n]["color"],
           markeredgecolor="white", markeredgewidth=1.0, linewidth=2.4, label=n)
    for n in ["Classical GBD", "Imitation Learning", "Proposed", "RL (Random Init)"]
]
leg = ax.legend(handles=handles, ncol=2, loc="lower left",
                frameon=False, fontsize=10, title_fontsize=11)
leg.get_title().set_fontweight("bold")
for t in leg.get_texts(): t.set_fontweight("bold")

# ====================== EXPORT DATA FOR SI ===================
os.makedirs("data/processed", exist_ok=True)

med_classical = agg_median_only("Classical GBD", sel_gbd)
med_il        = agg_median_only("Imitation Learning", sel_il)
med_prop      = agg_median_only("Proposed", sel_prop)
med_rl        = agg_median_only("RL (Random Init)", sel_rl)

Lmax = max(med_classical.size, med_il.size, med_prop.size, med_rl.size)
def to_len(a, L):
    a = np.asarray(a, dtype=float)
    return np.concatenate([a, np.full(L - a.size, np.nan)]) if a.size < L else a[:L]

df = pd.DataFrame({
    "iteration": np.arange(1, Lmax + 1, dtype=int),
    "Classical GBD":      to_len(med_classical, Lmax),
    "Imitation Learning": to_len(med_il, Lmax),
    "Proposed":           to_len(med_prop, Lmax),
    "RL (Random Init)":   to_len(med_rl, Lmax),
})
csv_path = "data/processed/figA_gap_curves.csv"
df.to_csv(csv_path, index=False)
print(f"[saved] {csv_path}  (rows={len(df)})")

last_row = {
    "Classical GBD":      med_classical[-1] if med_classical.size else np.nan,
    "Imitation Learning": med_il[-1]        if med_il.size        else np.nan,
    "Proposed":           med_prop[-1]      if med_prop.size      else np.nan,
    "RL (Random Init)":   med_rl[-1]        if med_rl.size        else np.nan,
}
pd.DataFrame([last_row]).to_csv("data/processed/figA_gap_last.csv", index=False)

fmt = lambda v: ("" if (pd.isna(v) or v is None) else f"{v:.3g}")
print("\n% ===== LaTeX table for SI (Figure A) =====")
print("\\begin{table}[h]")
print("\\centering")
print("\\caption{Median Benders gap by iteration for each method (Figure A). Raw gaps (not log-transformed).}")
print("\\small")
print("\\begin{tabular}{r r r r r}")
print("\\toprule")
print("\\textbf{Iter} & \\textbf{Classical} & \\textbf{IL} & \\textbf{Proposed} & \\textbf{RL (rand)}\\\\")
print("\\midrule")
for _, r in df.iterrows():
    print(f"{int(r['iteration'])} & {fmt(r['Classical GBD'])} & {fmt(r['Imitation Learning'])} & {fmt(r['Proposed'])} & {fmt(r['RL (Random Init)'])}\\\\")
print("\\bottomrule")
print("\\end{tabular}")
print("\\end{table}")

# ====================== SAVE FIGURE =========================
os.makedirs("final_plots", exist_ok=True)
plt.tight_layout()
plt.savefig("final_plots/gbd_gap_median_only.pdf", dpi=300, bbox_inches="tight")
if EXPORT_EPS:
    plt.savefig("final_plots/gbd_gap_median_only.eps", format="eps", bbox_inches="tight")
print("[saved] final_plots/gbd_gap_median_only.pdf")
if EXPORT_EPS:
    print("[saved] final_plots/gbd_gap_median_only.eps")
plt.show()
