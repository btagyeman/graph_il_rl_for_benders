import casadi as cs 
import numpy as np
from pyomo.environ import *




def build_master_model(self):
    model = ConcreteModel()
    model.y1 = Var(bounds=(0, 1), initialize=self.y_guess[0], within=Binary)
    model.y2 = Var(bounds=(0, 1), initialize=self.y_guess[1], within=Binary)
    model.y3 = Var(bounds=(0, 1), initialize=self.y_guess[2], within=Binary)
    model.y4 = Var(bounds=(0, 1), initialize=self.y_guess[3], within=Binary)
    model.y5 = Var(bounds=(0, 1), initialize=self.y_guess[4], within=Binary)
    model.mu = Var(bounds=(-1000, 1000), initialize=self.mu_guess)
    model.con_0 = Constraint(expr=model.y1 + model.y2 == self.rhs_y12)
    model.con_1 = Constraint(expr=model.y4 + model.y5 <= self.rhs_y45)

    if len(self.dual_vars_opt)> 0:
        for i in range(len(self.dual_vars_opt)):
            duals_opt = self.dual_vars_opt[i]
            x = self.non_comp_vars_opt[i]
            x3, x5, x9, x11, x13, x16 = x
            lag_opt = 0
            lag_opt += (self.coeff_y1*model.y1 + self.coeff_y2*model.y2 + self.coeff_y3*model.y3 +
                self.coeff_y4*model.y4 + self.coeff_y5*model.y5
                - 10*x3 - 15*x5 - 15*x9 + 15*x11 + 5*x13 - 20*x16 +
                np.exp(x3) + np.exp(x5/1.2) - 60*np.log(x11 + x13 + 1) + 140)
            lag_opt += duals_opt[0]*(-np.log(x11 + x13 + 1))
            lag_opt += duals_opt[1]*(-x3 - x5 - 2*x9 + x11 + 2*x16)
            lag_opt += duals_opt[2]*(-x3 - x5 - 0.75*x9 + x11 + 2*x16)
            lag_opt += duals_opt[3]*(x9 - x16)
            lag_opt += duals_opt[4]*(2*x9 - x11 - 2*x16)
            lag_opt += duals_opt[5]*(-0.5*x11 + x13)
            lag_opt += duals_opt[6]*(0.2*x11 - x13)
            lag_opt += duals_opt[7]*(np.exp(x3) - self.U*model.y1 - self.rhs_logical_1)
            lag_opt += duals_opt[8]*(np.exp(x5/1.2) - self.U*model.y2 - self.rhs_logical_2)
            lag_opt += duals_opt[9]*(1.25*x9 - self.U*model.y3)
            lag_opt += duals_opt[10]*(x11 + x13 - self.U*model.y4)
            lag_opt += duals_opt[11]*(-2*x9 + 2*x16 - self.U*model.y5)
            model.add_component(f'lag_opt_{i}', Constraint(expr=lag_opt - model.mu <= 0))

    if len(self.dual_vars_inf) > 0:
        for j in range(len(self.dual_vars_inf)):
            duals_inf = self.dual_vars_inf[j]
            x = self.non_comp_vars_inf[j]
            x3, x5, x9, x11, x13, x16 = x
            lag_inf = 0
            lag_inf += (self.coeff_y1*model.y1 + self.coeff_y2*model.y2 + self.coeff_y3*model.y3 +
                self.coeff_y4*model.y4 + self.coeff_y5*model.y5
                - 10*x3 - 15*x5 - 15*x9 + 15*x11 + 5*x13 - 20*x16 +
                np.exp(x3) + np.exp(x5/1.2) - 60*np.log(x11 + x13 + 1) + 140)
            lag_inf += duals_inf[0]*(-np.log(x11 + x13 + 1))
            lag_inf += duals_inf[1]*(-x3 - x5 - 2*x9 + x11 + 2*x16)
            lag_inf += duals_inf[2]*(-x3 - x5 - 0.75*x9 + x11 + 2*x16)
            lag_inf += duals_inf[3]*(x9 - x16)
            lag_inf += duals_inf[4]*(2*x9 - x11 - 2*x16)
            lag_inf += duals_inf[5]*(-0.5*x11 + x13)
            lag_inf += duals_inf[6]*(0.2*x11 - x13)
            lag_inf += duals_inf[7]*(np.exp(x3) - self.U*model.y1 - self.rhs_logical_1)
            lag_inf += duals_inf[8]*(np.exp(x5/1.2) - self.U*model.y2 - self.rhs_logical_2)
            lag_inf += duals_inf[9]*(1.25*x9 - self.U*model.y3)
            lag_inf += duals_inf[10]*(x11 + x13 - self.U*model.y4)
            lag_inf += duals_inf[11]*(-2*x9 + 2*x16 - self.U*model.y5)
            model.add_component(f'lag_inf_{j}', Constraint(expr=lag_inf <= 0))
    model.objective = Objective(expr=model.mu, sense=minimize)
    return model 

# def master_problem(self):
#    #print("Actions:", actions)
#     model = build_master_model(self)
#     solver = SolverFactory('gurobi')
#     results = solver.solve(model, tee=False)

#     if results.solver.termination_condition in [TerminationCondition.optimal, TerminationCondition.feasible]:
#         self.mp_feasibility = 'True'
#         self.LBD = value(model.mu)
#         self.y1, self.y2, self.y3, self.y4, self.y5 = (
#             value(model.y1), value(model.y2), value(model.y3), value(model.y4), value(model.y5))
#         self.modified_actions = [self.y1, self.y2, self.y3, self.y4, self.y5]
#         self.past_values = [self.y1, self.y2, self.y3, self.y4, self.y5]
#         self.y = [self.y1, self.y2, self.y3, self.y4, self.y5]
#         self.reward_time = results.solver.time

#     else:
#        # print("Master problem infeasible.")
#         self.mp_feasibility = 'False'
#         model = build_master_model(self)
#         solver = SolverFactory('gurobi')
#         results = solver.solve(model, tee=False)
#         self.reward_infs_mp = -1.50
#         self.LBD = value(model.mu)
        
#         self.y1, self.y2, self.y3, self.y4, self.y5 = (
#             value(model.y1), value(model.y2), value(model.y3), value(model.y4), value(model.y5))
#         self.modified_actions = [self.y1, self.y2, self.y3, self.y4, self.y5]
#         self.past_values = [self.y1, self.y2, self.y3, self.y4, self.y5]
#         self.y = [self.y1, self.y2, self.y3, self.y4, self.y5]

#     return

def master_problem_vanilla_pyomo(self):
    model = build_master_model(self)
    solver = SolverFactory('gurobi')
    results = solver.solve(model, tee=False)
    self.lbd = value(model.mu)
    self.y = [value(model.y1), value(model.y2), value(model.y3), value(model.y4), value(model.y5)]
    self.reward_time = results.solver.time
    self.mu = value(model.mu)
    self.mu_guess = value(model.mu)
    return