import casadi as cs 
import numpy as np
import contextlib
import os

@contextlib.contextmanager
def suppress_output():
    with open(os.devnull, 'w') as devnull:
        with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
            yield

def primal_problem_optimality(self, y_values, stage="inter_step"):
    w, lbw, ubw, G, lbg, ubg, Guess = [], [], [], [], [], [], []
    a = [0, 0, 0, 0, 0, 0]
    b = [2, 2, 2, np.inf, np.inf, 3]
    y_guess = self.y_guess
    x_guess = self.x_guess

    y = [cs.MX.sym(f'y_{i+1}') for i in range(5)]
    w += y
    lbw += y_values
    ubw += y_values
    Guess += y_guess

    x = [cs.MX.sym(f'x_{idx}') for idx in [3, 5, 9, 11, 13, 16]]
    w += x
    lbw += a
    ubw += b
    Guess += x_guess

    x3, x5, x9, x11, x13, x16 = x
    y1, y2, y3, y4, y5 = y

    G += [-cs.log(x11 + x13 + 1),
        -x3 - x5 - 2*x9 + x11 + 2*x16,
        -x3 - x5 - 0.75*x9 + x11 + 2*x16,
        x9 - x16,
        2*x9 - x11 - 2*x16,
        -0.5*x11 + x13,
        0.2*x11 - x13,
        cs.exp(x3) - self.U*y1 - self.rhs_logical_1,
        cs.exp(x5/1.2) - self.U*y2 - self.rhs_logical_2,
        1.25*x9 - self.U*y3,
        x11 + x13 - self.U*y4,
        -2*x9 + 2*x16 - self.U*y5]

    lbg += [-cs.inf] * len(G)
    ubg += [0] * len(G)

    J = (self.coeff_y1*y1 + self.coeff_y2*y2 + self.coeff_y3*y3 + self.coeff_y4*y4 + self.coeff_y5*y5
        - 10*x3 - 15*x5 - 15*x9 + 15*x11 + 5*x13 - 20*x16+ cs.exp(x3) + cs.exp(x5/1.2) - 60*cs.log(x11 + x13 + 1) + 140)

    nlp = {'x': cs.vertcat(*w), 'f': J, 'g': cs.vertcat(*G)}
    solver = cs.nlpsol('solver', 'ipopt', nlp, {'ipopt.print_level': 0, 'print_time': False, 'ipopt.sb': "yes"})
    sol = solver(lbx=lbw, ubx=ubw, lbg=lbg, ubg=ubg)

    opt_vals = sol['x'].full().ravel()
    obj_value = sol['f'].full().ravel()[0]
    solver_stats = solver.stats()['return_status']

    if solver_stats == 'Solve_Succeeded' and stage == 'inter_step':
        dual_vars = sol['lam_g'].full().ravel()
        self.dual_vars_opt.append(dual_vars.tolist())
        self.x = opt_vals[5:].tolist()
        self.x_guess = opt_vals[5:].tolist()
        self.non_comp_vars_opt.append(opt_vals[5:].tolist())
        self.ubd_cur = obj_value
        self.ubd = min(self.ubd_cur, self.ubd_prev)
        self.ubd_prev = self.ubd

    elif solver_stats == 'Solve_Succeeded' and stage == 'final_step':
        self.x = opt_vals[5:].tolist()
        self.final_cost = obj_value

    else:
        primal_problem_feasibility(self, y_values)


def primal_problem_feasibility(self, y_values):
    w, lbw, ubw, G, lbg, ubg, Guess = [], [], [], [], [], [], []
    a = [0, 0, 0, 0, 0, 0]
    b = [2, 2, 2, np.inf, np.inf, 3]
    y_guess = self.y_guess
    x_guess = self.x_guess

    y = [cs.MX.sym(f'y_{i+1}') for i in range(5)]
    w += y
    lbw += y_values
    ubw += y_values
    Guess += y_guess

    a_vars = [cs.MX.sym(f'a_{i+1}') for i in range(12)]
    w += a_vars
    lbw += [0]*12
    ubw += [np.inf]*12
    Guess += [1]*12

    x = [cs.MX.sym(f'x_{idx}') for idx in [3, 5, 9, 11, 13, 16]]
    w += x
    lbw += a
    ubw += b
    Guess += x_guess

    x3, x5, x9, x11, x13, x16 = x
    y1, y2, y3, y4, y5 = y

    G += [-cs.log(x11 + x13 + 1) - a_vars[0],
        -x3 - x5 - 2*x9 + x11 + 2*x16 - a_vars[1],
        -x3 - x5 - 0.75*x9 + x11 + 2*x16 - a_vars[2],
        x9 - x16 - a_vars[3],
        2*x9 - x11 - 2*x16 - a_vars[4],
        -0.5*x11 + x13 - a_vars[5],
        0.2*x11 - x13 - a_vars[6],
        cs.exp(x3) - self.U*y1 - self.rhs_logical_1 - a_vars[7],
        cs.exp(x5/1.2) - self.U*y2 - self.rhs_logical_2 - a_vars[8],
        1.25*x9 - self.U*y3 - a_vars[9],
        x11 + x13 - self.U*y4 - a_vars[10],
        -2*x9 + 2*x16 - self.U*y5 - a_vars[11]]

    lbg += [-np.inf] * len(G)
    ubg += [0] * len(G)
    J = cs.sum1(cs.vertcat(*a_vars))
    nlp = {'x': cs.vertcat(*w), 'f': J, 'g': cs.vertcat(*G)}
    solver = cs.nlpsol('solver', 'ipopt', nlp, {'ipopt.print_level': 0, 'print_time': False, 'ipopt.sb': "yes"})
    sol = solver(lbx=lbw, ubx=ubw, lbg=lbg, ubg=ubg)
    opt_vals = sol['x'].full().ravel()
    dual_vars = sol['lam_g'].full().ravel()
    self.dual_vars_inf.append(dual_vars.tolist())
    self.non_comp_vars_inf.append(opt_vals[17:].tolist())
    self.x_guess = opt_vals[17:].tolist()
    return