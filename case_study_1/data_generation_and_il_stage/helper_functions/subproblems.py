import casadi as cs
import contextlib
import os
import numpy as np

# Context manager to suppress stdout and stderr
@contextlib.contextmanager
def suppress_output():
    with open(os.devnull, 'w') as devnull:
        with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
            yield


def solve_final_problem(self):
    w = []
    lbw = []
    ubw = []
    J = 0
    G = []
    lbg = []
    ubg = []
    discrete = []
    Guess = []
    a = [0, 0, 0, 0, 0, 0]
    b = [2, 2, 2, np.inf, np.inf, 3]
    y_guess = self.y_guess
    x_guess = self.x_guess

    y = [cs.MX.sym(f'y_{i+1}') for i in range(5)]
    w.extend(y)
    lbw.extend([0]*5)
    ubw.extend([1]*5)
    discrete.extend([True]*5)
    Guess.extend(y_guess)

    x = [cs.MX.sym(f'x_{i}') for i in [3, 5, 9, 11, 13, 16]]
    x3, x5, x9, x11, x13, x16 = x
    y1, y2, y3, y4, y5 = y
    w.extend(x)
    lbw.extend([a[i] for i in range(6)])
    ubw.extend([b[i] for i in range(6)])
    discrete.extend([False]*6)
    Guess.extend(x_guess)

    G+=[-cs.log(x11 + x13 + 1),
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
        -2*x9 + 2*x16 - self.U*y5,
        y1 + y2 - self.rhs_y12,
        y4 + y5 - self.rhs_y45]

    ubg.extend([0]*len(G))
    lbg.extend([-cs.inf]*len(G))
    lbg[12] = 0 

    
    J = (self.coeff_y1*y1 + self.coeff_y2*y2 + self.coeff_y3*y3 + self.coeff_y4*y4 + self.coeff_y5*y5
        -10*x3 - 15*x5 - 15*x9 + 15*x11 + 5*x13 - 20*x16 + cs.exp(x3) + cs.exp(x5/1.2) - 60*cs.log(x11 + x13 + 1) + 140)


    minlp = {'x': cs.vertcat(*w), 'f': J, 'g': cs.vertcat(*G)}
    opts = {'discrete': discrete}

    with suppress_output():
        solver = cs.nlpsol('solver', 'bonmin', minlp, opts)
        sol = solver(x0 = Guess, lbx=lbw, ubx=ubw, lbg=lbg, ubg=ubg)

    self.x = sol['x'].full().ravel()[5:].tolist()
    self.y = sol['x'].full().ravel()[:5].tolist()
    return


def primal_problem_optimality(self, y_values, stage="inter_step"):
    w, lbw, ubw, G, lbg, ubg, Guess = [], [], [], [], [], [], []
    a = [0, 0, 0, 0, 0, 0]
    b = [2, 2, 2, np.inf, np.inf, 3]
    x_guess = self.x_guess

    y = [cs.MX.sym(f'y_{i+1}') for i in range(5)]
    w += y
    lbw += y_values
    ubw += y_values
    Guess += y_values

    x = [cs.MX.sym(f'x_{idx}') for idx in [3, 5, 9, 11, 13, 16]]
    w += x
    lbw += a
    ubw += b
    Guess += x_guess

    x3, x5, x9, x11, x13, x16 = x
    y1, y2, y3, y4, y5 = y
    # U = self.U

    # G += [-cs.log(x11 + x13 + 1),
    #     -x3 - x5 - 2*x9 + x11 + 2*x16,
    #     -x3 - x5 - 0.75*x9 + x11 + 2*x16,
    #     x9 - x16,
    #     2*x9 - x11 - 2*x16,
    #     -0.5*x11 + x13,
    #     0.2*x11 - x13,
    #     cs.exp(x3) - self.U*y1 - self.rhs_logical_1,
    #     cs.exp(x5/1.2) - self.U*y2 - self.rhs_logical_2,
    #     1.25*x9 - self.U*y3,
    #     x11 + x13 - self.U*y4,
    #     -2*x9 + 2*x16 - self.U*y5,
    #     y1 + y2 - self.rhs_y12,
    #     y4 + y5 - self.rhs_y45]

    # lbg += [-cs.inf] * len(G)
    # ubg += [0] * len(G)
    # lbg[12] = 0  # equality constraint for y1 + y2


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
        -2*x9 + 2*x16 - self.U*y5,
        y4 + y5 - self.rhs_y45]

    lbg += [-np.inf] * len(G)
    ubg += [0] * len(G)
    # lbg[12] = 0  # equality constraint for y1 + y2

    J = (self.coeff_y1*y1 + self.coeff_y2*y2 + self.coeff_y3*y3 +
        self.coeff_y4*y4 + self.coeff_y5*y5
        - 10*x3 - 15*x5 - 15*x9 + 15*x11 + 5*x13 - 20*x16
        + cs.exp(x3) + cs.exp(x5/1.2) - 60*cs.log(x11 + x13 + 1) + 140)

    nlp = {'x': cs.vertcat(*w), 'f': J, 'g': cs.vertcat(*G)}
    solver = cs.nlpsol('solver', 'ipopt', nlp, {'ipopt.print_level': 0, 'print_time': False, 'ipopt.sb': "yes"})
    # solver = cs.nlpsol('solver', 'ipopt', nlp, {'ipopt.print_level': 0, 'print_time': False, 'ipopt.sb': "yes",
    # 'ipopt.tol': 1e-10,  # or tighter
    # 'ipopt.constr_viol_tol': 1e-10,
    # 'ipopt.dual_inf_tol': 1e-10,
    # 'ipopt.acceptable_tol': 1e-10,
    # 'ipopt.acceptable_constr_viol_tol': 1e-10})
    sol = solver(lbx=lbw, ubx=ubw, lbg=lbg, ubg=ubg)

    opt_vals = sol['x'].full().ravel()
    obj_value = sol['f'].full().ravel()[0]
    solver_stats = solver.stats()['return_status']

    if solver_stats == 'Solve_Succeeded' and stage == 'inter_step':
        dual_vars = sol['lam_g'].full().ravel()
        # print(dual_vars)
        self.dual_vars_opt.append(dual_vars.tolist())
        self.x = opt_vals[5:].tolist()
        self.x_guess = opt_vals[5:].tolist()
        self.non_comp_vars_opt.append(opt_vals[5:].tolist())

        self.ubd_cur = obj_value
        self.ubd = min(self.ubd_cur, self.ubd_prev)
        self.ubd_prev = self.ubd
        self.cut_order.append('optimality_cut')

    elif solver_stats == 'Solve_Succeeded' and stage == 'final_step':
        self.x = opt_vals[5:].tolist()

    else:
        primal_problem_feasibility(self, y_values)


def primal_problem_feasibility(self, y_values):
    w, lbw, ubw, G, lbg, ubg, Guess = [], [], [], [], [], [], []
    a = [0, 0, 0, 0, 0, 0]
    b = [2, 2, 2, np.inf, np.inf, 3]

    # y_guess = self.y_guess
    x_guess = self.x_guess

    y = [cs.MX.sym(f'y_{i+1}') for i in range(5)]
    w += y
    lbw += y_values
    ubw += y_values
    Guess += y_values

    a_vars = [cs.MX.sym(f'a_{i+1}') for i in range(13)]
    w += a_vars
    lbw += [0] * 13
    ubw += [cs.inf] * 13
    Guess += [1] * 13

    x = [cs.MX.sym(f'x_{idx}') for idx in [3, 5, 9, 11, 13, 16]]
    w += x
    lbw += a
    ubw += b
    Guess += x_guess

    x3, x5, x9, x11, x13, x16 = x
    y1, y2, y3, y4, y5 = y
    # G += [
    #     -cs.log(x11 + x13 + 1) - a_vars[0],
    #     -x3 - x5 - 2*x9 + x11 + 2*x16 - a_vars[1],
    #     -x3 - x5 - 0.75*x9 + x11 + 2*x16 - a_vars[2],
    #     x9 - x16 - a_vars[3],
    #     2*x9 - x11 - 2*x16 - a_vars[4],
    #     -0.5*x11 + x13 - a_vars[5],
    #     0.2*x11 - x13 - a_vars[6],
    #     cs.exp(x3) - U*y1 - self.rhs_logical_1 - a_vars[7],
    #     cs.exp(x5/1.2) - U*y2 - self.rhs_logical_2 - a_vars[8],
    #     1.25*x9 - U*y3 - a_vars[9],
    #     x11 + x13 - U*y4 - a_vars[10],
    #     -2*x9 + 2*x16 - U*y5 - a_vars[11],
    #     y1 + y2 - self.rhs_y12,
    #     y4 + y5 - self.rhs_y45 - a_vars[12]
    # ]

    # lbg += [-cs.inf] * len(G)
    # ubg += [0] * len(G)
    # lbg[12] = 0  # equality constraint for y1 + y2

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
        -2*x9 + 2*x16 - self.U*y5 - a_vars[11],
        y4 + y5 - self.rhs_y45 - a_vars[12]]

    lbg += [-cs.inf] * len(G)
    ubg += [0] * len(G)
    # lbg[12] = 0  # equality constraint for y1 + y2

    J = cs.sum1(cs.vertcat(*a_vars))

    nlp = {'x': cs.vertcat(*w), 'f': J, 'g': cs.vertcat(*G)}
    solver = cs.nlpsol('solver', 'ipopt', nlp, {'ipopt.print_level': 0, 'print_time': False, 'ipopt.sb': "yes"})
    # solver = cs.nlpsol('solver', 'ipopt', nlp, {'ipopt.print_level': 0, 'print_time': False, 'ipopt.sb': "yes",
    # 'ipopt.tol': 1e-10,  # or tighter
    # 'ipopt.constr_viol_tol': 1e-10,
    # 'ipopt.dual_inf_tol': 1e-10,
    # 'ipopt.acceptable_tol': 1e-10,
    # 'ipopt.acceptable_constr_viol_tol': 1e-10})
    sol = solver(lbx=lbw, ubx=ubw, lbg=lbg, ubg=ubg)

    opt_vals = sol['x'].full().ravel()
    dual_vars = sol['lam_g'].full().ravel()
    # print(dual_vars)
    self.dual_vars_inf.append(dual_vars.tolist())
    self.non_comp_vars_inf.append(opt_vals[18:].tolist())
    self.x_guess = opt_vals[18:].tolist()
    self.cut_order.append('feasibility_cut')
    return