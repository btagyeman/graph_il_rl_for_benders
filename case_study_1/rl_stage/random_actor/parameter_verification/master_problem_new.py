import casadi as cs 
import contextlib
import os
import sys

# Context manager to suppress stdout and stderr
@contextlib.contextmanager
def suppress_output():
    with open(os.devnull, 'w') as devnull:
        with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
            yield
   
def master_problem(self):
    w, lbw, ubw, G, lbg, ubg, discrete, Guess = [], [], [], [], [], [], [], []
    y_vars = [cs.MX.sym(f'y_{i+1}') for i in range(5)]
    w += y_vars
    lbw += [0]*5
    ubw += [1]*5
    discrete+=[True]*5
    Guess+=self.y_guess

    mu = cs.MX.sym('mu')
    w.append(mu)
    lbw.append(-1000)
    ubw.append(1000)
    discrete.append(False)
    Guess.append(self.mu_guess)

    # Add optimality cuts
    if len(self.dual_vars_opt) > 0:
        for i in range(len(self.dual_vars_opt)):
            duals = self.dual_vars_opt[i]
            x = self.non_comp_vars_opt[i]
            x3, x5, x9, x11, x13, x16 = x
            y1, y2, y3, y4, y5 = y_vars

            lag_opt = 0
            lag_opt += (self.coeff_y1*y1 + self.coeff_y2*y2 + self.coeff_y3*y3 + self.coeff_y4*y4 + self.coeff_y5*y5
                -10*x3 - 15*x5 - 15*x9 + 15*x11 + 5*x13 - 20*x16 + cs.exp(x3) + cs.exp(x5/1.2) - 60*cs.log(x11 + x13 + 1) + 140)

            lag_opt += duals[0]*(-cs.log(x11 + x13 + 1))
            lag_opt += duals[1]*(-x3 - x5 - 2*x9 + x11 + 2*x16)
            lag_opt += duals[2]*(-x3 - x5 - 0.75*x9 + x11 + 2*x16)
            lag_opt += duals[3]*(x9 - x16)
            lag_opt += duals[4]*(2*x9 - x11 - 2*x16)
            lag_opt += duals[5]*(-0.5*x11 + x13)
            lag_opt += duals[6]*(0.2*x11 - x13)
            lag_opt += duals[7]*(cs.exp(x3) - self.U*y1 - self.rhs_logical_1)
            lag_opt += duals[8]*(cs.exp(x5/1.2) - self.U*y2 - self.rhs_logical_2)
            lag_opt += duals[9]*(1.25*x9 - self.U*y3)
            lag_opt += duals[10]*(x11 + x13 - self.U*y4)
            lag_opt += duals[11]*(-2*x9 + 2*x16 - self.U*y5)

            G.append(lag_opt - mu)
            lbg.append(-cs.inf)
            ubg.append(0)
            

    if len(self.dual_vars_inf) > 0:
        for i in range(len(self.dual_vars_inf)):
            duals = self.dual_vars_inf[i]
            x = self.non_comp_vars_inf[i]
            x3, x5, x9, x11, x13, x16 = x
            y1, y2, y3, y4, y5 = y_vars

            lag_inf = 0
            lag_inf += duals[0]*(-cs.log(x11 + x13 + 1))
            lag_inf += duals[1]*(-x3 - x5 - 2*x9 + x11 + 2*x16)
            lag_inf += duals[2]*(-x3 - x5 - 0.75*x9 + x11 + 2*x16)
            lag_inf += duals[3]*(x9 - x16)
            lag_inf += duals[4]*(2*x9 - x11 - 2*x16)
            lag_inf += duals[5]*(-0.5*x11 + x13)
            lag_inf += duals[6]*(0.2*x11 - x13)
            lag_inf += duals[7]*(cs.exp(x3) - self.U*y1 - self.rhs_logical_1)
            lag_inf += duals[8]*(cs.exp(x5/1.2) - self.U*y2 - self.rhs_logical_2)
            lag_inf += duals[9]*(1.25*x9 - self.U*y3)
            lag_inf += duals[10]*(x11 + x13 - self.U*y4)
            lag_inf += duals[11]*(-2*x9 + 2*x16 - self.U*y5)

            G.append(lag_inf)
            lbg.append(-cs.inf)
            ubg.append(0)


    # Add logical constraints
    y1, y2, y3, y4, y5 = y_vars
    G.append(y1 + y2 - self.rhs_y12)
    lbg.append(0)
    ubg.append(0)

    G.append(y4 + y5 - self.rhs_y45)
    lbg.append(-cs.inf)
    ubg.append(0)

    J = mu
    minlp = {'x': cs.vertcat(*w), 'f': J, 'g': cs.vertcat(*G)}
    opts = {'discrete': discrete}
    with suppress_output():
        solver = cs.nlpsol('solver', 'bonmin', minlp, opts)
        sol = solver(lbx=lbw, ubx=ubw, lbg=lbg, ubg=ubg)

    vals = sol['x'].full().ravel()
    self.y = vals[:5].tolist()
    self.y_1, self.y_2, self.y_3, self.y_4, self.y_5 = self.y 
    self.y_guess = vals[:5].tolist()
    self.mu = vals[5]
    self.mu_guess = vals[5]
    self.lbd = sol['f'].full().ravel()[0]
    return