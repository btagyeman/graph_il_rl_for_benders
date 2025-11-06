
from parameter_verification.master_problem_new import master_problem
from parameter_verification.subproblems_new import primal_problem_optimality

class GBDProblem:
    def __init__(self, ubd, lbd, mu_guess, x_guess, y_guess, feasible_pars):
        self.x = x_guess 
        self.y = y_guess 
        self.y_guess = y_guess
        self.y_1, self.y_2, self.y_3, self.y_4, self.y_5 = y_guess
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

        self.final_cost = 0

        self.coeff_y1 = feasible_pars[0]
        self.coeff_y2 = feasible_pars[1]
        self.coeff_y3 = feasible_pars[2]
        self.coeff_y4 = feasible_pars[3]
        self.coeff_y5 = feasible_pars[4]
        self.rhs_logical_1 = feasible_pars[5]
        self.rhs_logical_2 = feasible_pars[6]
        self.rhs_y12 = feasible_pars[7]
        self.rhs_y45 = feasible_pars[8]
        self.U = feasible_pars[9]

    def solve_master_problem(self):
        return master_problem(self)

    def solve_primal_problem(self, stage="inter_step"):
        return primal_problem_optimality(self, self.y, stage=stage)
    

def solve_gbd_problem(self, feasible_pars,ubd, lbd, mu_guess, x_guess, y_guess, epsilon = 0.001):
    gbd = GBDProblem(ubd, lbd, mu_guess, x_guess, y_guess, feasible_pars)
    while gbd.ubd - gbd.lbd > epsilon:
        gbd.solve_primal_problem(stage="inter_step")
        if gbd.ubd - gbd.lbd > epsilon:
            gbd.solve_master_problem()
        else:
            pass
        pass
    gbd.solve_primal_problem(stage="final_step")
    self.cost_gbd = gbd.final_cost
    return
