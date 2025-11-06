import casadi as cs
import contextlib
import os
import sys
import numpy as np

# Context manager to suppress stdout and stderr
@contextlib.contextmanager
def suppress_output():
    with open(os.devnull, 'w') as devnull:
        with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
            yield

def solve_initial_problem(self):
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
    y_guess = self.initial_guesses[0]
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
    U = self.U
    w.extend(x)
    lbw.extend([a[i] for i in range(6)])
    ubw.extend([b[i] if b[i] is not None else cs.inf for i in range(6)])
    discrete.extend([False]*6)
    Guess.extend(x_guess)


    G+=[-cs.log(x11 + x13 + 1),
        -x3 - x5 - 2*x9 + x11 + 2*x16,
        -x3 - x5 - 0.75*x9 + x11 + 2*x16,
        x9 - x16,
        2*x9 - x11 - 2*x16,
        -0.5*x11 + x13,
        0.2*x11 - x13,
        cs.exp(x3) - U*y1 - self.rhs_logical_1,
        cs.exp(x5/1.2) - U*y2 - self.rhs_logical_2,
        1.25*x9 - U*y3,
        x11 + x13 - U*y4,
        -2*x9 + 2*x16 - U*y5,
        y1 + y2 - self.rhs_y12,
        y4 + y5 - self.rhs_y45]

    ubg.extend([0]*len(G))
    lbg.extend([-np.inf]*len(G))
    lbg[12] = 0  # equality constraint for y1 + y2

    
    J = (self.coeff_y1*y1 + self.coeff_y2*y2 + self.coeff_y3*y3 + self.coeff_y4*y4 + self.coeff_y5*y5
        -10*x3 - 15*x5 - 15*x9 + 15*x11 + 5*x13 - 20*x16 + cs.exp(x3) + cs.exp(x5/1.2) - 60*cs.log(x11 + x13 + 1) + 140)


    minlp = {'x': cs.vertcat(*w), 'f': J, 'g': cs.vertcat(*G)}
    opts = {'discrete': discrete}

    with suppress_output():
        solver = cs.nlpsol('solver', 'bonmin', minlp, opts)
        sol = solver(x0 = Guess,lbx=lbw, ubx=ubw, lbg=lbg, ubg=ubg)

    stats = solver.stats()
    if stats['success']:
        solution = sol['x'].full().ravel().tolist()
        self.optimal_solutions.append(solution)
        
        self.feasible_parameters.append([self.coeff_y1,self.coeff_y2,self.coeff_y3,self.coeff_y4,self.coeff_y5,self.rhs_logical_1,self.rhs_logical_2,self.rhs_y12,self.rhs_y45,self.U])

    else:
        print("No feasible solution found for the current parameters.")
        pass 


def solve_initial_problem_comp(self, initial_guess, current_pars):
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
    y_guess = initial_guess
    x_guess = [1,1,1,1,1,1] 

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

    coeff_y1 = current_pars[0]
    coeff_y2 = current_pars[1]
    coeff_y3 = current_pars[2]
    coeff_y4 = current_pars[3]
    coeff_y5 = current_pars[4]
    rhs_logical_1 = current_pars[5]
    rhs_logical_2 = current_pars[6]
    rhs_y12 = current_pars[7]
    rhs_y45 = current_pars[8]
    U = current_pars[9]

    G+=[-cs.log(x11 + x13 + 1),
        -x3 - x5 - 2*x9 + x11 + 2*x16,
        -x3 - x5 - 0.75*x9 + x11 + 2*x16,
        x9 - x16,
        2*x9 - x11 - 2*x16,
        -0.5*x11 + x13,
        0.2*x11 - x13,
        cs.exp(x3) - U*y1 - rhs_logical_1,
        cs.exp(x5/1.2) - U*y2 - rhs_logical_2,
        1.25*x9 - U*y3,
        x11 + x13 - U*y4,
        -2*x9 + 2*x16 - U*y5,
        y1 + y2 - rhs_y12,
        y4 + y5 - rhs_y45]

    ubg.extend([0]*len(G))
    lbg.extend([-cs.inf]*len(G))
    lbg[12] = 0 

    
    J = (coeff_y1*y1 + coeff_y2*y2 + coeff_y3*y3 + coeff_y4*y4 + coeff_y5*y5
        -10*x3 - 15*x5 - 15*x9 + 15*x11 + 5*x13 - 20*x16 + cs.exp(x3) + cs.exp(x5/1.2) - 60*cs.log(x11 + x13 + 1) + 140)


    minlp = {'x': cs.vertcat(*w), 'f': J, 'g': cs.vertcat(*G)}
    opts = {'discrete': discrete}
    with suppress_output():
        solver = cs.nlpsol('solver', 'bonmin', minlp, opts)
        sol = solver(x0 = Guess, lbx=lbw, ubx=ubw, lbg=lbg, ubg=ubg)

    self.cost_orig_prob = sol['f'].full().item()
    return