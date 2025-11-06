from pyomo.environ import *
from parameter_verification.solve_initial_problem import solve_initial_problem_comp
from parameter_verification.solve_with_gbd import solve_gbd_problem

class EvaluateParameterSet():
    def __init__(self, initial_guess, par_dict):
        self.pars = par_dict
        self.initial_guess = initial_guess
        self.x_guess = [1,1,1,1,1,1]
        self.lbd = -1e5
        self.ubd = 1e5
        self.mu_guess = 0.6
        self.epsilon = 0.001
        self.use_parameters = None
        self.cost_orig_prob = 0
        self.cost_gbd = 0
        pass 

        #original parameters
        self.U = 10
        self.coeff_y1 = 5
        self.coeff_y2 = 8
        self.coeff_y3 = 6
        self.coeff_y4 = 10
        self.coeff_y5 = 6
        self.rhs_logical_1 = 1
        self.rhs_logical_2 = 1
        self.rhs_y12 = 1
        self.rhs_y45 = 1

    def set_parameters(self):
        self.coeff_y1 = self.pars[0]
        self.coeff_y2 = self.pars[1]
        self.coeff_y3 = self.pars[2]
        self.coeff_y4 = self.pars[3]
        self.coeff_y5 = self.pars[4]
        self.rhs_logical_1 = self.pars[5]
        self.rhs_logical_2 = self.pars[6]
        self.rhs_y12 = self.pars[7]
        self.rhs_y45 = self.pars[8]
        self.U = self.pars[9]
        pass

    def evaluate_parameter_set(self): # This method is automatically called by the constructor of Dataset class
        self.set_parameters()
        solve_initial_problem_comp(self,self.initial_guess,self.pars)
        solve_gbd_problem(self,self.pars,self.ubd,self.lbd,self.mu_guess,self.x_guess,self.initial_guess,epsilon=self.epsilon)

        if abs(self.cost_gbd - self.cost_orig_prob) < 0.0003:
            self.use_parameters = True
        else:
            self.use_parameters = False
            pass
        return
