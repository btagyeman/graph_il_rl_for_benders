from pyomo.environ import *
import logging
from helper_functions.graph_representation import create_graph_representation
from helper_functions.master_problems import master_problem_il, master_problem_rl
from helper_functions.primal_problems import primal_problem, initial_primal_problem
from spektral.data.loaders import DisjointLoader
from spektral.data import Dataset
import numpy as np
logging.getLogger('pyomo.core').setLevel(logging.ERROR)

class GraphListDataset(Dataset):
    def __init__(self, graph_list, **kwargs):
        self.graph_list = graph_list
        super().__init__(**kwargs)
    def read(self):
        return self.graph_list
    
class GBDENV:
    def __init__(self, UBD, LBD, feas_pars, x_guess, y_guess, epsilon=0.01, max_iter=100):
        assert UBD >= LBD, "Upper bound must be greater or equal to the lower bound"
        self.UBD_cur=UBD
        self.UBD_prev=UBD
        self.UBD = UBD
        self.LBD = LBD
        self.epsilon = epsilon
        self.max_iter = max_iter
        self.feas_pars = feas_pars
        self.x_guess = x_guess
        self.y_guess = y_guess
        self.mu_guess = 0.6
        self.x = x_guess
        self.y = y_guess
        self.y_il = y_guess
        self.y1, self.y2, self.y3, self.y4, self.y5 = y_guess
        # self.reward_infs_sp = 0
        self.reward_infs_mp = 0
        self.reward_fs_mp = 0
        self.reward_il = 0
        self.reward_time = 0
        self.reward_bnds = 0

        self.reward_lbd = 0
        self.reward_ubd = 0
        self.previous_gap = self.UBD - self.LBD
        self.initial_gap = self.UBD - self.LBD
        # self.lbd_prev = self.cd si    
        # self.lbd_cur = self.LBD
        # self.lbd_init = self.LBD

        # self.ubd_prev = self.UBD
        # self.ubd_cur = self.UBD
        # self.ubd_init = self.UBD

        self.modified_action = [self.y1, self.y2, self.y3, self.y4, self.y5]
        self.y1_il, self.y2_il, self.y3_il, self.y4_il, self.y5_il = self.y_il

        self.mp_feasibility = None
        self.n_variables = len(y_guess) # number of complicating variables
        self.dual_vars_opt=[]
        self.dual_vars_inf=[]
        self.non_comp_vars_opt=[]
        self.non_comp_vars_inf=[]

        self.coeff_y1 = self.feas_pars[0]
        self.coeff_y2 = self.feas_pars[1]
        self.coeff_y3 = self.feas_pars[2]
        self.coeff_y4 = self.feas_pars[3]
        self.coeff_y5 = self.feas_pars[4]
        self.rhs_logical_1 = self.feas_pars[5]
        self.rhs_logical_2 = self.feas_pars[6]
        self.rhs_y12 = self.feas_pars[7]
        self.rhs_y45 = self.feas_pars[8]
        self.U = self.feas_pars[9]

        self.graph_features = {}
        self.graph_features['optimality_cuts']={}
        self.graph_features['feasibility_cuts']={}
        self.cut_order = [] 
        self.optimal_values = [] 
        self.past_values = [self.y1, self.y2, self.y3, self.y4, self.y5] 
        self.predicted_binary_values = [] 
        self.predicted_values = []

    def solve_initial_primal_problem(self):
        initial_primal_problem(self)

    def solve_primal_problem(self, stage='Intermediate Step'):
        primal_problem(self, stage)

    def generate_graph_and_evaluate_agent(self):
        if len(self.dual_vars_opt) > 0:
            for i in range(len(self.dual_vars_opt)):
                duals_opt = self.dual_vars_opt[i]
                x = self.non_comp_vars_opt[i]
                x3, x5, x9, x11, x13, x16 = x

                # coeff_y1_opt = self.coeff_y1 - duals_opt[7]*self.U + duals_opt[12]
                # coeff_y2_opt = self.coeff_y2 - duals_opt[8]*self.U + duals_opt[12]
                # coeff_y3_opt = self.coeff_y3 - duals_opt[9]*self.U
                # coeff_y4_opt = self.coeff_y4 - duals_opt[10]*self.U + duals_opt[13]
                # coeff_y5_opt = self.coeff_y5 - duals_opt[11]*self.U + duals_opt[13]


                coeff_y1_opt = self.coeff_y1 - (duals_opt[7]*self.U)
                coeff_y2_opt = self.coeff_y2 - (duals_opt[8]*self.U)
                coeff_y3_opt = self.coeff_y3 - (duals_opt[9]*self.U)
                coeff_y4_opt = self.coeff_y4 - (duals_opt[10]*self.U)
                coeff_y5_opt = self.coeff_y5 - (duals_opt[11]*self.U)

                # constant_term_opt = - 10*x3 - 15*x5 - 15*x9 + 15*x11 + 5*x13 - 20*x16 +\
                #     np.exp(x3) + np.exp(x5/1.2) - 60*np.log(x11 + x13 + 1) + 140+\
                #     duals_opt[0]*(-np.log(x11 + x13 + 1)) + \
                #     duals_opt[1]*(-x3 - x5 - 2*x9 + x11 + 2*x16) + \
                #     duals_opt[2]*(-x3 - x5 - 0.75*x9 + x11 + 2*x16) + \
                #     duals_opt[3]*(x9 - x16) + \
                #     duals_opt[4]*(2*x9 - x11 - 2*x16) + \
                #     duals_opt[5]*(-0.5*x11 + x13) + \
                #     duals_opt[6]*(0.2*x11 - x13) + \
                #     duals_opt[7]*(np.exp(x3) -self.rhs_logical_1) + \
                #     duals_opt[8]*(np.exp(x5/1.2)-self.rhs_logical_2) + \
                #     duals_opt[9]*(1.25*x9 ) + \
                #     duals_opt[10]*(x11 + x13 ) + \
                #     duals_opt[11]*(-2*x9 + 2*x16) + \
                #     duals_opt[12]*(-self.rhs_y12)   + \
                #     duals_opt[13]*(-self.rhs_y45)



                constant_term_opt = - 10*x3 - 15*x5 - 15*x9 + 15*x11 + 5*x13 - 20*x16 +\
                    np.exp(x3) + np.exp(x5/1.2) - 60*np.log(x11 + x13 + 1) + 140+\
                    duals_opt[0]*(-np.log(x11 + x13 + 1)) + \
                    duals_opt[1]*(-x3 - x5 - 2*x9 + x11 + 2*x16) + \
                    duals_opt[2]*(-x3 - x5 - 0.75*x9 + x11 + 2*x16) + \
                    duals_opt[3]*(x9 - x16) + \
                    duals_opt[4]*(2*x9 - x11 - 2*x16) + \
                    duals_opt[5]*(-0.5*x11 + x13) + \
                    duals_opt[6]*(0.2*x11 - x13) + \
                    duals_opt[7]*(np.exp(x3) -self.rhs_logical_1) + \
                    duals_opt[8]*(np.exp(x5/1.2)-self.rhs_logical_2) + \
                    duals_opt[9]*(1.25*x9 ) + \
                    duals_opt[10]*(x11 + x13 ) + \
                    duals_opt[11]*(-2*x9 + 2*x16) 

                self.graph_features['optimality_cuts'][i+1] = (coeff_y1_opt,coeff_y2_opt,coeff_y3_opt,coeff_y4_opt,coeff_y5_opt,-constant_term_opt)
                                   
        if len(self.dual_vars_inf) > 0:
            for j in range(len(self.dual_vars_inf)):
                duals_inf = self.dual_vars_inf[j]
                x = self.non_comp_vars_inf[j]
                x3, x5, x9, x11, x13, x16 = x

                # coeff_y1_inf = -duals_inf[7]*self.U  + duals_inf[12]
                # coeff_y2_inf = -duals_inf[8]*self.U  + duals_inf[12]
                # coeff_y3_inf = -duals_inf[9]*self.U
                # coeff_y4_inf = -duals_inf[10]*self.U + duals_inf[13]
                # coeff_y5_inf = -duals_inf[11]*self.U + duals_inf[13]

                coeff_y1_inf = -duals_inf[7]*self.U  
                coeff_y2_inf = -duals_inf[8]*self.U 
                coeff_y3_inf = -duals_inf[9]*self.U
                coeff_y4_inf = -duals_inf[10]*self.U
                coeff_y5_inf = -duals_inf[11]*self.U

                # constant_term_inf =  duals_inf[0]*(-np.log(x11 + x13 + 1)) + \
                #     duals_inf[1]*(-x3 - x5 - 2*x9 + x11 + 2*x16) + \
                #     duals_inf[2]*(-x3 - x5 - 0.75*x9 + x11 + 2*x16) + \
                #     duals_inf[3]*(x9 - x16) + \
                #     duals_inf[4]*(2*x9 - x11 - 2*x16) + \
                #     duals_inf[5]*(-0.5*x11 + x13) + \
                #     duals_inf[6]*(0.2*x11 - x13) + \
                #     duals_inf[7]*(np.exp(x3) - self.rhs_logical_1) + \
                #     duals_inf[8]*(np.exp(x5/1.2) - self.rhs_logical_2) + \
                #     duals_inf[9]*(1.25*x9) + \
                #     duals_inf[10]*(x11 + x13) + \
                #     duals_inf[11]*(-2*x9 + 2*x16) + \
                #     duals_inf[12]*(-self.rhs_y12) + \
                #     duals_inf[13]*(-self.rhs_y45)
                
                constant_term_inf =  duals_inf[0]*(-np.log(x11 + x13 + 1)) + \
                    duals_inf[1]*(-x3 - x5 - 2*x9 + x11 + 2*x16) + \
                    duals_inf[2]*(-x3 - x5 - 0.75*x9 + x11 + 2*x16) + \
                    duals_inf[3]*(x9 - x16) + \
                    duals_inf[4]*(2*x9 - x11 - 2*x16) + \
                    duals_inf[5]*(-0.5*x11 + x13) + \
                    duals_inf[6]*(0.2*x11 - x13) + \
                    duals_inf[7]*(np.exp(x3) - self.rhs_logical_1) + \
                    duals_inf[8]*(np.exp(x5/1.2) - self.rhs_logical_2) + \
                    duals_inf[9]*(1.25*x9) + \
                    duals_inf[10]*(x11 + x13) + \
                    duals_inf[11]*(-2*x9 + 2*x16)
            
                self.graph_features['feasibility_cuts'][j+1] = (coeff_y1_inf,coeff_y2_inf,coeff_y3_inf,coeff_y4_inf,coeff_y5_inf,-constant_term_inf)


        current_graph = create_graph_representation(self.cut_order,self.graph_features,self.past_values,self.past_values,self.n_variables)
        loader = DisjointLoader(GraphListDataset([current_graph]), batch_size=1, epochs=1)
        observations_graph, _ = next(iter(loader))
        return observations_graph
    
    def solve_master_problem_il(self):
        master_problem_il(self)
        return

    def solve_master_problem_rl(self, actions):
        master_problem_rl(self, actions)
        return

    def compute_rewards(self):
        if self.mp_feasibility == 'True':
            # # Compute a reward that relates to the lowver bound, since the actions of the agent are related to the lower bound 
            # delta_lbd = self.LBD - self.lbd_prev
            # norm_lbd = abs(self.lbd_init)
            # self.reward_lbd = delta_lbd/norm_lbd
            # self.lbd_prev = self.LBD

            # # Compute a reward that indicates the effect of the agents actions on the upper bound
            # delta_ubd = abs(self.UBD - self.ubd_prev)
            # norm_ubd = abs(self.ubd_init)
            # self.reward_ubd = delta_ubd/norm_ubd
            # # print(self.reward_ubd)
            # self.ubd_prev = self.UBD

            current_gap = self.UBD - self.LBD
            # self.reward_bnds = (self.previous_gap - current_gap)/self.previous_gap # reward is the difference between the previous gap and the current gap
            self.reward_bnds = (self.previous_gap - current_gap)/self.initial_gap # reward is the difference between the previous gap and the current gap
            # print(self.reward_bnds)
            # self.reward_bnds = (self.previous_gap - current_gap)
            self.previous_gap = current_gap

            # The agent is able to solve the master problem and get a feasible solution, so the reward is the difference between the upper and lower bounds
            # self.reward_bnds = (self.UBD - self.LBD)/self.UBD
            self.reward_il = abs(self.y1_il-self.y1) + abs(self.y2_il-self.y2) + abs(self.y3_il-self.y3) + abs(self.y4_il-self.y4) + abs(self.y5_il-self.y5)
            self.reward_infs_mp = 0.50
            # self.reward_fs_mp = 0.50
            # self.reward_time = -np.tanh(self.reward_time)
            # print(self.reward_time)
        else:
            # self.reward_lbd = 0
            # self.reward_ubd = 0
            self.reward_bnds = 0 # if the master problem is infeasible, the reward is 0
            self.reward_il = 0 # if the master problem is infeasible, the reward is 0
            # self.reward_time = 0
            # self.reward_infs_mp = -1.5
            # self.reward_fs_mp =  0

        # total_reward = -self.reward_bnds - self.reward_il + self.reward_infs_sp + self.reward_infs_mp + self.reward_time
        # total_reward = self.reward_bnds - self.reward_il + self.reward_infs_sp + self.reward_infs_mp + self.reward_time

        self.reward_time = -min(self.reward_time, 1)
        # total_reward = 0*4*self.reward_lbd + 1*3*self.reward_ubd - 0*self.reward_il + self.reward_infs_mp + 1*self.reward_time + self.reward_fs_mp
        # total_reward = 1*4*self.reward_lbd + 1*3*self.reward_ubd - 0*1.0*self.reward_il + self.reward_infs_mp + 1*self.reward_time + self.reward_fs_mp
        total_reward = 6*self.reward_bnds - 3*self.reward_il + 1*self.reward_time + self.reward_infs_mp

        # print(total_reward)
        return total_reward
    
    def step(self, actions):
        # 1. RL Agent's assignment → solves master problem
        self.solve_master_problem_rl(actions)

        # 2. IL agent baseline → for reward_il component
        self.solve_master_problem_il()

        # 3. Subproblem → solves x and adds cuts
        self.solve_primal_problem(stage='Intermediate Step')

        # 4. Graph representation → input for next observation
        obs = self.generate_graph_and_evaluate_agent()
        # print(self.LBD)

        reward = self.compute_rewards()
        modified_action = self.modified_action
        done = (self.UBD - self.LBD) <= self.epsilon or len(self.cut_order) >= self.max_iter
        info = {"LBD": self.LBD,"UBD": self.UBD,"feasible": self.mp_feasibility,"cuts": len(self.cut_order)}
        return obs, reward, done, info, modified_action
    
    def reset(self):
        self.solve_initial_primal_problem()
        return self.generate_graph_and_evaluate_agent()
