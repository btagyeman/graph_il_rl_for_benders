import numpy as np 
import casadi as cs 
import time
from helper_functions.master_problem_vanilla import master_problem
from helper_functions.primal_problems_vanilla import primal_problem_optimality


class GBDProblem:
    def __init__(self, ubd, lbd, mu_guess, x_guess, y_guess):
        self.x = x_guess 
        self.y = y_guess 
        self.y_guess = y_guess
        self.x_guess= x_guess
        self.dual_vars_opt = []
        self.dual_vars_inf = []
        self.non_comp_vars_opt = []
        self.non_comp_vars_inf = []
        self.ubd = ubd 
        self.lbd = lbd
        self.ubd_cur = ubd
        self.ubd_prev = ubd
        self.mu = mu_guess
        self.mu_guess = mu_guess

    
    def set_parameters(self, par_dict):
        self.coeff_y1 = par_dict[0]
        self.coeff_y2 = par_dict[1]
        self.coeff_y3 = par_dict[2]
        self.coeff_y4 = par_dict[3]
        self.coeff_y5 = par_dict[4]
        self.rhs_logical_1 = par_dict[5]
        self.rhs_logical_2 = par_dict[6]
        self.rhs_y12 = par_dict[7]
        self.rhs_y45 = par_dict[8]
        self.U = par_dict[9]
        return

    def solve_master_problem(self):
        return master_problem(self)

    def solve_primal_problem(self, stage="inter_step"):
        return primal_problem_optimality(self, self.y, stage=stage)
    

def gbd_algorithm(feasible_pars,ubd, lbd, mu_guess, x_guess, y_guess, epsilon = 0.001):
    gbd = GBDProblem(ubd, lbd, mu_guess, x_guess, y_guess)
    gbd.set_parameters(feasible_pars)
    lbds = []
    ubds = []

    time_mp = 0    

    lbds.append(gbd.lbd)
    ubds.append(gbd.ubd)
    while gbd.ubd - gbd.lbd > epsilon:

        gbd.solve_primal_problem(stage="inter_step")
        ubds.append(gbd.ubd)
        if gbd.ubd - gbd.lbd > epsilon:
            time_mp_start = time.time()
            gbd.solve_master_problem()
            time_mp_end = time.time()
            time_mp += (time_mp_end - time_mp_start)
            lbds.append(gbd.lbd)
            pass 
        pass 

    gbd.solve_primal_problem(stage="final_step")

    if len(lbds) < len(ubds):
        lbds.append(gbd.lbd)
    elif len(ubds) < len(lbds):
        ubds.append(gbd.ubd)
    return gbd.x, gbd.y, gbd.lbd, lbds, ubds, time_mp


# def gbd_algorithm(feasible_pars,ubd, lbd, mu_guess, x_guess, y_guess, epsilon = 0.001):
#     gbd = GBDProblem(ubd, lbd, mu_guess, x_guess, y_guess)
#     gbd.set_parameters(feasible_pars)
#     lbds = []
#     ubds = []

#     time_mp = 0    

#     lbds.append(gbd.lbd)
#     ubds.append(gbd.ubd)
#     while gbd.ubd - gbd.lbd > epsilon:

#         gbd.solve_primal_problem(stage="inter_step")
#         ubds.append(gbd.ubd)

#         time_mp_start = time.time()
#         gbd.solve_master_problem()
#         time_mp_end = time.time()
#         time_mp += (time_mp_end - time_mp_start)
#         lbds.append(gbd.lbd)


#     gbd.solve_primal_problem(stage="final_step")

#     # Ensure equal lengths (pad if needed)
#     # if len(lbds) < len(ubds):
#     #     lbds.append(gbd.lbd)
#     # elif len(ubds) < len(lbds):
#     #     ubds.append(gbd.ubd)
#     # print("The final bounds are: ", gbd.lbd, gbd.ubd)
#     # opt_soln = []
#     # opt_soln.extend(gbd.y)
#     # opt_soln.extend(gbd.x)
#     return gbd.x, gbd.y, gbd.lbd, lbds, ubds, time_mp


def solve_problem_with_gbd(initial_guess, feasible_pars):
    x_guess = [1,1,1,1,1,1]
    mu_guess = 0.6
    ubd =  100000
    lbd = -100000
    x, y,lbd_final, lbds,  ubds, time_mp = gbd_algorithm(feasible_pars, ubd, lbd, mu_guess, x_guess, initial_guess)
    return x, y, lbd_final, lbds, ubds, time_mp

