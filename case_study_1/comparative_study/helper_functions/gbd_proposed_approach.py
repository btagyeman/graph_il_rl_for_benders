from pyomo.environ import *
import numpy as np
from helper_functions.graph_representation import create_graph_representation
from helper_functions.primal_problems import primal_problem, initial_primal_problem
from helper_functions.master_problems import  master_problem
from spektral.data.loaders import DisjointLoader
from spektral.layers import GlobalSumPool, ECCConv
from tensorflow.keras.layers import Dense
from tensorflow.keras import Model
from spektral.data import Dataset
import tensorflow as tf
import time


x_mean = np.load('./scaling_data/x_mean.npy')
x_std = np.load('./scaling_data/x_std.npy')
e_mean = np.load('./scaling_data/e_mean.npy')
e_std = np.load('./scaling_data/e_std.npy')

epsilon = 1e-8

class GraphListDataset(Dataset):
    def __init__(self, graph_list, **kwargs):
        self.graph_list = graph_list
        super().__init__(**kwargs)
    def read(self):
        return self.graph_list

    
class IL_AGENT(Model):
    def __init__(self):
        super().__init__()
        self.conv1 = ECCConv(64, activation='relu')
        self.conv2 = ECCConv(64, activation='relu')
        self.pool = GlobalSumPool()
        self.dense1 = Dense(64, activation='relu')
        self.out_y1 = Dense(1)
        self.out_y2 = Dense(1)
        self.out_y3 = Dense(1)
        self.out_y4 = Dense(1)
        self.out_y5 = Dense(1)

    def call(self, inputs):
        x, a, e, i = inputs
        x = (x-x_mean)/(x_std + epsilon)
        e = (e-e_mean)/(e_std + epsilon)
        x = self.conv1([x, a, e])
        x = self.conv2([x, a, e])
        x = self.pool([x, i])
        x = self.dense1(x)
        y1 = self.out_y1(x)
        y2 = self.out_y2(x)
        y3 = self.out_y3(x)
        y4 = self.out_y4(x)
        y5 = self.out_y5(x)
        return tf.concat([y1, y2, y3, y4, y5], axis=-1)
    
# class IL_AGENT(Model):
#     def __init__(self):
#         super().__init__()
#         # self.norm = BatchNormalization()  # Normalize node features
#         self.conv1 = ECCConv(64, activation='relu')
#         self.conv2 = ECCConv(64, activation='relu')
#         self.conv3 = ECCConv(64, activation='relu')
#         self.pool = GlobalSumPool()
#         self.dense1 = Dense(64, activation='relu')
#         self.dense2 = Dense(32, activation='relu')
#         # self.dense3 = Dense(16, activation='relu')
#         self.out_y1 = Dense(1)
#         self.out_y2 = Dense(1)
#         self.out_y3 = Dense(1)
#         self.out_y4 = Dense(1)
#         self.out_y5 = Dense(1)

#     def call(self, inputs):
#         x, a, e, i = inputs
#         # x = self.norm(x)  # Batch normalization on node features
#         x = (x-x_mean)/(x_std + epsilon)
#         e = (e-e_mean)/(e_std + epsilon)
#         x = self.conv1([x, a, e])
#         x = self.conv2([x, a, e])
#         x = self.conv3([x, a, e])
#         x = self.pool([x, i])
#         x = self.dense1(x)
#         x = self.dense2(x)
#         y1 = self.out_y1(x)
#         y2 = self.out_y2(x)
#         y3 = self.out_y3(x)
#         y4 = self.out_y4(x)
#         y5 = self.out_y5(x)
#         return tf.concat([y1, y2, y3, y4, y5], axis=-1)
    
agent = IL_AGENT()
    
class GBDENV:
    def __init__(self, UBD, LBD, feas_pars, x_guess, y_guess, epsilon,agent_type):
        assert UBD >= LBD, "Upper bound must be greater or equal to the lower bound"
        self.UBD_cur=UBD
        self.UBD_prev=UBD
        self.UBD = UBD
        self.LBD = LBD
        self.LBD_cur  = LBD
        self.LBD_prev =  LBD

        self.UBD_act = UBD
        # # Load the agent weights based on the agent_type
        # if agent_type == 'rl_il_init':
        #     agent.load_weights("./saved_rl_agents_il/rl_agent_logits_il_30_episodes/variables/variables")
        # elif agent_type == 'rl_random_init':
        #     agent.load_weights("./saved_rl_agents_rand/rl_agent_logits_rand_30_episodes/variables/variables")
        # else:
        #     agent.load_weights("./gbd_agents/mh_il_agent_logits/variables/variables")

        # Load the agent weights based on the agent_type
        if agent_type == 'rl_il_init':
            agent.load_weights("./gbd_agents/best_agent_proposed/variables/variables").expect_partial()
        elif agent_type == 'rl_random_init':
            agent.load_weights("./gbd_agents/best_agent_rand_actor/variables/variables").expect_partial()
        else:
            agent.load_weights("./gbd_agents/best_agent_il/variables/variables").expect_partial()


        self.epsilon = epsilon
        # self.max_iter = max_iter
        self.feas_pars = feas_pars
        self.x_guess = x_guess
        self.y_guess = y_guess
        self.mu_guess = 0.6
        self.x = x_guess
        self.y = y_guess
        self.y1, self.y2, self.y3, self.y4, self.y5 = y_guess

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
        # print(self.cut_order)
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
                

                coeff_y1_opt = self.coeff_y1 - (duals_opt[7]*self.U)
                coeff_y2_opt = self.coeff_y2 - (duals_opt[8]*self.U)
                coeff_y3_opt = self.coeff_y3 - (duals_opt[9]*self.U)
                coeff_y4_opt = self.coeff_y4 - (duals_opt[10]*self.U)
                coeff_y5_opt = self.coeff_y5 - (duals_opt[11]*self.U)

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

                coeff_y1_inf = -duals_inf[7]*self.U  
                coeff_y2_inf = -duals_inf[8]*self.U 
                coeff_y3_inf = -duals_inf[9]*self.U
                coeff_y4_inf = -duals_inf[10]*self.U
                coeff_y5_inf = -duals_inf[11]*self.U

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

        with tf.device('/CPU:0'):
            y_pred = agent(observations_graph, training=False).numpy().squeeze()
            y_pred = tf.sigmoid(y_pred).numpy().squeeze()  # Apply sigmoid activation to the output
            pass 
    

        self.predicted_values.append(y_pred)

        THRESH_HIGH = 0.90
        THRESH_LOW = 0.10
        # print(y_pred)

        # if self.UBD - self.LBD > 5:
        #     THRESH_HIGH = 0.80
        #     THRESH_LOW = 0.20
        # elif 1 <= self.UBD - self.LBD <= 5:
        #     THRESH_HIGH = 0.90
        #     THRESH_LOW = 0.10
        # elif 0.1 <= self.UBD - self.LBD < 1:
        #     THRESH_HIGH = 0.95
        #     THRESH_LOW = 0.05
        # else:
        #     THRESH_HIGH = 1.0001
        #     THRESH_LOW = -0.00001

        binary_pred = np.where(y_pred >= THRESH_HIGH, 1, np.where(y_pred <= THRESH_LOW, 0, -1))

        if binary_pred[0] != -1 and binary_pred[1] != -1:
            if (binary_pred[0] + binary_pred[1]) != self.rhs_y12:
                binary_pred[0] = -1
                binary_pred[1] = -1

        if binary_pred[3] != -1 and binary_pred[4] != -1:
            if (binary_pred[3] + binary_pred[4]) > self.rhs_y45:
                binary_pred[3] = -1
                binary_pred[4] = -1
        # binary_pred = [-1,-1,-1,-1,-1]
        self.y1 = binary_pred[0]
        self.y2 = binary_pred[1]
        self.y3 = binary_pred[2]
        self.y4 = binary_pred[3]
        self.y5 = binary_pred[4]
        self.predicted_binary_values.append(binary_pred)
        self.y = [self.y1, self.y2, self.y3, self.y4, self.y5]  
        return
    
    
    # def compute_bounds_full_assignment(self):
    #     ## TO DO: Modify this function.
    #     bounds = []

    #         # Optimality cuts
    #     if len(self.dual_vars_opt) > 0:
    #         for i in range(len(self.dual_vars_opt)):
    #             duals_opt = self.dual_vars_opt[i]
    #             x = self.non_comp_vars_opt[i]
    #             x3, x5, x9, x11, x13, x16 = x
    #             y1, y2, y3, y4, y5 = self.y1, self.y2, self.y3, self.y4, self.y5
    #             lag_opt = 0

    #             lag_opt += (self.coeff_y1*y1 + self.coeff_y2*y2 + self.coeff_y3*y3 +
    #             self.coeff_y4*y4 + self.coeff_y5*y5
    #             - 10*x3 - 15*x5 - 15*x9 + 15*x11 + 5*x13 - 20*x16 +
    #             np.exp(x3) + np.exp(x5/1.2) - 60*np.log(x11 + x13 + 1) + 140
    #             )

    #             lag_opt += duals_opt[0]*(-np.log(x11 + x13 + 1))
    #             lag_opt += duals_opt[1]*(-x3 - x5 - 2*x9 + x11 + 2*x16)
    #             lag_opt += duals_opt[2]*(-x3 - x5 - 0.75*x9 + x11 + 2*x16)
    #             lag_opt += duals_opt[3]*(x9 - x16)
    #             lag_opt += duals_opt[4]*(2*x9 - x11 - 2*x16)
    #             lag_opt += duals_opt[5]*(-0.5*x11 + x13)
    #             lag_opt += duals_opt[6]*(0.2*x11 - x13)
    #             lag_opt += duals_opt[7]*(np.exp(x3) - self.U*y1 - self.rhs_logical_1)
    #             lag_opt += duals_opt[8]*(np.exp(x5/1.2) - self.U*y2 - self.rhs_logical_2)
    #             lag_opt += duals_opt[9]*(1.25*x9 - self.U*y3)
    #             lag_opt += duals_opt[10]*(x11 + x13 - self.U*y4)
    #             lag_opt += duals_opt[11]*(-2*x9 + 2*x16 - self.U*y5)
    #             lag_opt += duals_opt[12]* (y1 + y2 - self.rhs_y12)
    #             lag_opt += duals_opt[13]*(y4 + y5 - self.rhs_y45)

    #             bounds.append(lag_opt)
    #             pass 
    #         bounds = np.array(bounds)
    #         self.LBD = bounds.max()
    #     else:
    #         self.LBD = -1000
    #         pass 
    #     self.past_values = [self.y1, self.y2, self.y3, self.y4, self.y5]

    def compute_bounds_full_assignment(self):
        ## TO DO: Modify this function.
        bounds = []

            # Optimality cuts
        if len(self.dual_vars_opt) > 0:
            for i in range(len(self.dual_vars_opt)):
                duals_opt = self.dual_vars_opt[i]
                x = self.non_comp_vars_opt[i]
                x3, x5, x9, x11, x13, x16 = x
                y1, y2, y3, y4, y5 = self.y1, self.y2, self.y3, self.y4, self.y5
                lag_opt = 0

                lag_opt += (self.coeff_y1*y1 + self.coeff_y2*y2 + self.coeff_y3*y3 +
                self.coeff_y4*y4 + self.coeff_y5*y5
                - 10*x3 - 15*x5 - 15*x9 + 15*x11 + 5*x13 - 20*x16 +
                np.exp(x3) + np.exp(x5/1.2) - 60*np.log(x11 + x13 + 1) + 140
                )

                lag_opt += duals_opt[0]*(-np.log(x11 + x13 + 1))
                lag_opt += duals_opt[1]*(-x3 - x5 - 2*x9 + x11 + 2*x16)
                lag_opt += duals_opt[2]*(-x3 - x5 - 0.75*x9 + x11 + 2*x16)
                lag_opt += duals_opt[3]*(x9 - x16)
                lag_opt += duals_opt[4]*(2*x9 - x11 - 2*x16)
                lag_opt += duals_opt[5]*(-0.5*x11 + x13)
                lag_opt += duals_opt[6]*(0.2*x11 - x13)
                lag_opt += duals_opt[7]*(np.exp(x3) - self.U*y1 - self.rhs_logical_1)
                lag_opt += duals_opt[8]*(np.exp(x5/1.2) - self.U*y2 - self.rhs_logical_2)
                lag_opt += duals_opt[9]*(1.25*x9 - self.U*y3)
                lag_opt += duals_opt[10]*(x11 + x13 - self.U*y4)
                lag_opt += duals_opt[11]*(-2*x9 + 2*x16 - self.U*y5)

                bounds.append(lag_opt)
                pass 
            bounds = np.array(bounds)
            # self.LBD = bounds.max()

            # self.LBD_cur = bounds.max()
            self.LBD_cur = bounds.max().item()
            LBD_temp = max(self.LBD_cur, self.LBD_prev)

            # if self.LBD_cur > self.UBD:
            if LBD_temp > self.UBD:
            # if abs(LBD_temp - self.UBD) > 1e-3:
                self.y1 = -1
                self.y2 = -1
                self.y3 = -1
                self.y4 = -1
                self.y5 = -1
                self.y = [self.y1, self.y2, self.y3, self.y4, self.y5]
                self.solve_master_problem(self.y)
            else:
                # self.LBD = max(self.LBD_cur, self.LBD_prev)
                # self.LBD_prev = self.LBD
                self.LBD = LBD_temp
                self.LBD_prev = self.LBD
                self.past_values = [self.y1, self.y2, self.y3, self.y4, self.y5]
            # self.LBD = max(self.LBD_cur, self.LBD_prev)
            # self.LBD_prev = self.LBD
        else:
            # self.LBD = -1000
            self.LBD_cur = -1000
            self.LBD = max(self.LBD_cur, self.LBD_prev)
            self.LBD_prev = self.LBD
            self.past_values = [self.y1, self.y2, self.y3, self.y4, self.y5]
            pass 
        

    # def evaluate_master_problem(self):
    #     comp_vars = [self.y1, self.y2, self.y3, self.y4, self.y5]
    #     comp_vars = np.array(comp_vars)

    #     if np.any(comp_vars == -1):
    #         self.solve_master_problem(self.y)

    #     else:
    #         if ((comp_vars[0] + comp_vars[1]) - self.rhs_y12) > 1e-6 or comp_vars[3] + comp_vars[4] > self.rhs_y45:
    #             print(comp_vars[0] + comp_vars[1] - self.rhs_y12)
    #             print(comp_vars[3] + comp_vars[4] - self.rhs_y45)
    #             print("Full assignment not feasible, solving master problem")
    #             self.y1 = -1
    #             self.y2 = -1
    #             self.y3 = -1
    #             self.y4 = -1
    #             self.y5 = -1
    #             self.y = [self.y1, self.y2, self.y3, self.y4, self.y5]
    #             self.solve_master_problem(self.y)
    #         else:
    #             self.compute_bounds_full_assignment()
    #         # self.compute_bounds_full_assignment()
    #         pass

    def evaluate_master_problem(self):
        comp_vars = [self.y1, self.y2, self.y3, self.y4, self.y5]
        comp_vars = np.array(comp_vars)

        if np.any(comp_vars == -1):
            self.solve_master_problem(self.y)

        else:
            self.compute_bounds_full_assignment()
            # self.compute_bounds_full_assignment()
            pass  

    # def evaluate_master_problem(self):
    #     self.solve_master_problem(self.y)


    # def evaluate_master_problem(self):
    #     comp_vars = [self.y1, self.y2, self.y3, self.y4, self.y5]
    #     comp_vars = np.array(comp_vars)

    #     if np.any(comp_vars == -1):
    #         self.solve_master_problem(self.y)

    #     else:
    #         if ((comp_vars[0] + comp_vars[1]) - self.rhs_y12) > 1e-6 or comp_vars[3] + comp_vars[4] > self.rhs_y45:
    #             print(comp_vars[0] + comp_vars[1] - self.rhs_y12)
    #             print(comp_vars[3] + comp_vars[4] - self.rhs_y45)
    #             print("Full assignment not feasible, solving master problem")
    #             self.y1 = -1
    #             self.y2 = -1
    #             self.y3 = -1
    #             self.y4 = -1
    #             self.y5 = -1
    #             self.y = [self.y1, self.y2, self.y3, self.y4, self.y5]
    #             self.solve_master_problem(self.y)
    #         else:
    #             self.compute_bounds_full_assignment()
    #         # self.compute_bounds_full_assignment()
    #         pass 

    # def evaluate_master_problem(self):
    #     actions = self.y
    #     # comp_vars = [self.y1, self.y2, self.y3, self.y4, self.y5]
    #     # comp_vars = np.array(comp_vars)

    #     # if np.any(comp_vars == -1):
    #     self.solve_master_problem(actions)
        # else:
            # self.compute_bounds_full_assignment()
            # pass 

    def solve_master_problem(self, actions):
        # print(actions)
        master_problem(self, actions)
        return
    
    def set_new_lbds(self, lbd):
        self.LBD  = lbd

    def set_new_lbds(self, new_lbd_candidate, epsilon=1e-4):
        self.LBD = min(self.UBD - epsilon, max(self.LBD_prev, new_lbd_candidate))
        self.LBD_prev = self.LBD

        return self.LBD


# def gbd_algorithm(UBD, LBD, feas_pars, x_guess, y_guess, agent_type, eps_):
#     lbds_proposed = []
#     ubds_proposed = []
#     gbd = GBDENV(UBD, LBD, feas_pars, x_guess, y_guess, eps_, agent_type)

#     lbds_proposed.append(gbd.LBD)
#     ubds_proposed.append(gbd.UBD)

#     # First Phase – initial GBD loop
#     while gbd.UBD - gbd.LBD > eps_:
#         gbd.solve_primal_problem(stage="Intermediate Step")
#         ubds_proposed.append(gbd.UBD)
#         if gbd.UBD - gbd.LBD > eps_:
#             gbd.generate_graph_and_evaluate_agent()
#             gbd.evaluate_master_problem()
#             lbds_proposed.append(gbd.LBD)

#     # Terminal check
#     if abs(gbd.UBD - gbd.LBD) <= eps_:
#         gbd.solve_primal_problem(stage="Final Step")
#     else:
#         # Start recursive correction
#         gbd.set_new_lbds(gbd.LBD_cur)
#         recursive_gbd_correction(gbd, lbds_proposed, ubds_proposed, eps_)

#     print(gbd.UBD)
#     print(gbd.LBD_cur)
#     print(gbd.LBD)
#     return gbd.x, gbd.y, gbd.predicted_binary_values, gbd.predicted_values, gbd.LBD, lbds_proposed, ubds_proposed

# def gbd_algorithm(UBD, LBD, feas_pars, x_guess, y_guess, agent_type, eps_):
#     lbds_proposed = []
#     ubds_proposed = []
#     gbd = GBDENV(UBD, LBD, feas_pars, x_guess, y_guess, eps_, agent_type)

#     lbds_proposed.append(gbd.LBD)
#     ubds_proposed.append(gbd.UBD)

#     converged = False
#     restart_attempts = 0
#     max_restarts = 10  # Safety limit

#     while not converged and restart_attempts <= max_restarts:
#         while gbd.UBD - gbd.LBD > eps_:
#             gbd.solve_primal_problem(stage="Intermediate Step")
#             ubds_proposed.append(gbd.UBD)

#             if gbd.UBD - gbd.LBD > eps_:
#                 gbd.generate_graph_and_evaluate_agent()
#                 gbd.evaluate_master_problem()
#                 lbds_proposed.append(gbd.LBD)

#         # Check convergence
#         if abs(gbd.UBD - gbd.LBD) <= eps_:
#             gbd.solve_primal_problem(stage="Final Step")
#             converged = True
#         else:
#             # Restart needed
#             gbd.set_new_lbds(gbd.LBD_cur)
#             restart_attempts += 1
#             print(f"[Warning] Restarting GBD: attempt {restart_attempts} due to LBD > UBD")

#     if not converged:
#         print("[Error] GBD failed to converge after max restarts. Final LBD still exceeds UBD.")

#     print("Final UBD:", gbd.UBD)
#     print("Final LBD_cur:", gbd.LBD_cur)
#     print("Final LBD:", gbd.LBD)
#     return gbd.x, gbd.y, gbd.predicted_binary_values, gbd.predicted_values, gbd.LBD, lbds_proposed, ubds_proposed

# def gbd_algorithm(UBD, LBD, feas_pars, x_guess, y_guess, agent_type, eps_):
#     lbds_proposed = []
#     ubds_proposed = []

#     gbd = GBDENV(UBD, LBD, feas_pars, x_guess, y_guess, eps_, agent_type)

#     lbds_proposed.append(gbd.LBD)
#     ubds_proposed.append(gbd.UBD)

#     converged = False
#     restart_attempts = 0
#     max_restarts = 10  # Prevent infinite loops

#     while not converged and restart_attempts <= max_restarts:
#         while gbd.UBD - gbd.LBD > eps_:
#             gbd.solve_primal_problem(stage="Intermediate Step")
#             ubds_proposed.append(gbd.UBD)

#             if gbd.UBD - gbd.LBD > eps_:
#                 gbd.generate_graph_and_evaluate_agent()
#                 gbd.evaluate_master_problem()
#                 lbds_proposed.append(gbd.LBD)

#         # ---- Enhanced Convergence Handling ---- #
#         if abs(gbd.UBD - gbd.LBD) <= eps_:
#             gbd.solve_primal_problem(stage="Final Step")
#             converged = True

#         elif restart_attempts < max_restarts:
#             gbd.set_new_lbds(gbd.LBD_cur)
#             restart_attempts += 1
#             print(f"[Warning] Restarting GBD: attempt {restart_attempts} due to LBD > UBD")

#         else:
#             print("[Fallback] Solving full master problem to recover feasibility...")
#             gbd.evaluate_master_problem(actions=None)
#             gbd.solve_primal_problem(stage="Final Step")
#             converged = True
#         # --------------------------------------- #

#     if not converged:
#         print("[Error] GBD failed to converge after max restarts. Final LBD still exceeds UBD.")

#     print("Final UBD:", gbd.UBD)
#     # print("Final LBD_cur:", gbd.LBD_cur)
#     print("Final LBD:", gbd.LBD)

#     return gbd.x, gbd.y, gbd.predicted_binary_values, gbd.predicted_values, gbd.LBD, lbds_proposed, ubds_proposed


def gbd_algorithm(UBD, LBD, feas_pars, x_guess, y_guess, agent_type, eps_):
    lbds_proposed = []
    ubds_proposed = []
    gbd = GBDENV(UBD, LBD, feas_pars, x_guess, y_guess, eps_, agent_type)

    lbds_proposed.append(gbd.LBD)
    ubds_proposed.append(gbd.UBD)

    converged = False
    restart_attempts = 0
    max_restarts = 10  # Safety limit
    time_mp = 0

    while not converged and restart_attempts <= max_restarts:
        # ───── Regular GBD Loop ─────
        while gbd.UBD - gbd.LBD > eps_:
            gbd.solve_primal_problem(stage="Intermediate Step")
            ubds_proposed.append(gbd.UBD)

            if gbd.UBD - gbd.LBD > eps_:
                gbd.generate_graph_and_evaluate_agent()
                time_mp_start = time.time()
                gbd.evaluate_master_problem()
                time_mp_end = time.time()
                time_mp += (time_mp_end - time_mp_start)
                lbds_proposed.append(gbd.LBD)

        # ───── Check Termination ─────
        # if abs(gbd.UBD - gbd.LBD) <= eps_:
        if abs(gbd.UBD - gbd.LBD) <= 0.01:
            gbd.solve_primal_problem(stage="Final Step")
            converged = True

        # elif round(gbd.LBD,2) > round(gbd.UBD,2) + eps_:
        # elif gbd.LBD > gbd.UBD + eps_:
        elif gbd.LBD > gbd.UBD + 0.01:
        
        # elif gbd.LBD > gbd.UBD + 0.01:
            # ───── LBD Exceeded UBD: Reset and Resume GBD ─────
            lbd_new = gbd.set_new_lbds(gbd.LBD_cur)
            restart_attempts += 1
            # print(f"[Warning] Restarting GBD: attempt {restart_attempts} due to LBD > UBD")
            # lbds_proposed.pop()
            # lbds_proposed.append(lbd_new)
            
        

            # Solve full master problem without fixing any binary variables
            # gbd.evaluate_master_problem(actions=None)
            time_mp_start = time.time()
            gbd.solve_master_problem(actions=None)
            # lbds_proposed.append(gbd.LBD)
            time_mp_end = time.time()
            time_mp += (time_mp_end - time_mp_start)

            # Resume GBD loop using accumulated cuts
            while gbd.UBD - gbd.LBD > eps_:
                gbd.solve_primal_problem(stage="Intermediate Step")
                ubds_proposed.append(gbd.UBD)

                if gbd.UBD - gbd.LBD > eps_:
                    gbd.generate_graph_and_evaluate_agent()
                    time_mp_start = time.time()
                    gbd.evaluate_master_problem()
                    time_mp_end = time.time()
                    time_mp += (time_mp_end - time_mp_start)
                    lbds_proposed.append(gbd.LBD)

            # if abs(gbd.UBD - gbd.LBD) <= eps_:
            if abs(gbd.UBD - gbd.LBD) <= 0.01:
                gbd.solve_primal_problem(stage="Final Step")
                converged = True

    # # ───── Final Check ─────
    # if not converged:
    #     print("[Error] GBD failed to converge after max restarts. Final LBD still exceeds UBD.")

    # print("Final UBD:", gbd.UBD)
    # print("Final LBD_cur:", gbd.LBD_cur)
    # print("Final LBD:", gbd.LBD)

    if len(lbds_proposed) < len(ubds_proposed):
        lbds_proposed.append(gbd.LBD)
    elif len(ubds_proposed) < len(lbds_proposed):
        ubds_proposed.append(gbd.UBD)

    return (
        gbd.x,
        gbd.y,
        gbd.predicted_binary_values,
        gbd.predicted_values,
        gbd.LBD,
        lbds_proposed,
        ubds_proposed, time_mp
    )

def gbd_algorithm_new(UBD, LBD, feas_pars, x_guess, y_guess, agent_type, eps_):
    lbds_proposed = []
    ubds_proposed = []
    gbd = GBDENV(UBD, LBD, feas_pars, x_guess, y_guess, eps_, agent_type)

    lbds_proposed.append(gbd.LBD)
    ubds_proposed.append(gbd.UBD)

    time_mp = 0.0


    while gbd.UBD - gbd.LBD > eps_:
        gbd.solve_primal_problem(stage="Intermediate Step")
        ubds_proposed.append(gbd.UBD)


        gbd.generate_graph_and_evaluate_agent()
        time_mp_start = time.time()
        gbd.evaluate_master_problem()
        time_mp_end = time.time()
        time_mp += (time_mp_end - time_mp_start)
        lbds_proposed.append(gbd.LBD)

        # ───── Check Termination ─────
        # if abs(gbd.UBD - gbd.LBD) <= eps_:
    if abs(gbd.UBD - gbd.LBD) <= 0.01:
        gbd.solve_primal_problem(stage="Final Step")

    return (
        gbd.x,
        gbd.y,
        gbd.predicted_binary_values,
        gbd.predicted_values,
        gbd.LBD,
        lbds_proposed,
        ubds_proposed, time_mp
    )
# def gbd_algorithm(UBD, LBD, feas_pars, x_guess, y_guess, agent_type, eps_):
#     lbds_proposed = []
#     ubds_proposed = []
#     gbd = GBDENV(UBD, LBD, feas_pars, x_guess, y_guess, eps_, agent_type)

#     lbds_proposed.append(gbd.LBD)
#     ubds_proposed.append(gbd.UBD)

#     converged = False
#     restart_attempts = 0
#     max_restarts = 10  # Safety limit
#     time_mp = 0.0

#     while not converged and restart_attempts <= max_restarts:
#         # ───── Regular GBD Loop ─────
#         while gbd.UBD - gbd.LBD > eps_:
#             gbd.solve_primal_problem(stage="Intermediate Step")
#             ubds_proposed.append(gbd.UBD)

#             if gbd.UBD - gbd.LBD > eps_:
#                 gbd.generate_graph_and_evaluate_agent()
#                 time_mp_start = time.time()
#                 gbd.evaluate_master_problem()
#                 time_mp_end = time.time()
#                 time_mp += (time_mp_end - time_mp_start)
#                 lbds_proposed.append(gbd.LBD)

#         # ───── Check Termination ─────
#         if abs(gbd.UBD - gbd.LBD) <= 0.01:
#             gbd.solve_primal_problem(stage="Final Step")
#             converged = True

#         elif gbd.LBD > gbd.UBD + 0.01:
#             # ───── LBD Exceeded UBD: Reset and Resume GBD ─────
#             # Last LBD is invalid (overshoot) → remove it (keep sequences aligned)
#             if len(lbds_proposed) > 0:
#                 lbds_proposed.pop()
#                 if len(ubds_proposed) > len(lbds_proposed):
#                     ubds_proposed.pop()

#             gbd.set_new_lbds(gbd.LBD_cur)
#             restart_attempts += 1
#             print(f"[Warning] Restarting GBD: attempt {restart_attempts} due to LBD > UBD")

#             # Solve full master problem without fixing any binary variables
#             time_mp_start = time.time()
#             gbd.solve_master_problem(actions=None)
#             time_mp_end = time.time()
#             time_mp += (time_mp_end - time_mp_start)
#             # Append the restored, valid LBD
#             lbds_proposed.append(gbd.LBD)

#             # Resume GBD loop using accumulated cuts
#             while gbd.UBD - gbd.LBD > eps_:
#                 gbd.solve_primal_problem(stage="Intermediate Step")
#                 ubds_proposed.append(gbd.UBD)

#                 if gbd.UBD - gbd.LBD > eps_:
#                     gbd.generate_graph_and_evaluate_agent()
#                     time_mp_start = time.time()
#                     gbd.evaluate_master_problem()
#                     time_mp_end = time.time()
#                     time_mp += (time_mp_end - time_mp_start)
#                     lbds_proposed.append(gbd.LBD)

#             if abs(gbd.UBD - gbd.LBD) <= 0.01:
#                 gbd.solve_primal_problem(stage="Final Step")
#                 converged = True

#     # ───── Final Check ─────
#     if not converged:
#         print("[Error] GBD failed to converge after max restarts. Final LBD still exceeds UBD.")

#     # Ensure sequences have equal length for plotting (pad the shorter with its terminal value)
#     if len(lbds_proposed) < len(ubds_proposed):
#         lbds_proposed.append(gbd.LBD)
#     elif len(ubds_proposed) < len(lbds_proposed):
#         ubds_proposed.append(gbd.UBD)

#     print("Final UBD:", gbd.UBD)
#     print("Final LBD_cur:", gbd.LBD_cur)
#     print("Final LBD:", gbd.LBD)

#     return (
#         gbd.x,
#         gbd.y,
#         gbd.predicted_binary_values,
#         gbd.predicted_values,
#         gbd.LBD,
#         lbds_proposed,
#         ubds_proposed,
#         time_mp,
#     )


def recursive_gbd_correction(gbd, lbds_proposed, ubds_proposed, eps_):
    while gbd.UBD - gbd.LBD > eps_:
        gbd.solve_primal_problem(stage="Intermediate Step")
        ubds_proposed.append(gbd.UBD)
        if gbd.UBD - gbd.LBD > eps_:
            gbd.generate_graph_and_evaluate_agent()
            gbd.evaluate_master_problem()
            lbds_proposed.append(gbd.LBD)

    # Terminal check
    if abs(gbd.UBD - gbd.LBD) <= eps_:
        gbd.solve_primal_problem(stage="Final Step")
    else:
        # Lower bound exceeded UBD again; reset and recurse
        gbd.set_new_lbds(gbd.LBD_cur)
        recursive_gbd_correction(gbd, lbds_proposed, ubds_proposed, eps_)



# def gbd_algorithm(UBD,LBD,feas_pars,x_guess,y_guess,agent_type,eps_):
#     lbds_proposed = []
#     ubds_proposed = []
#     gbd = GBDENV(UBD,LBD,feas_pars,x_guess,y_guess,eps_,agent_type)
#     # print(gbd.UBD)
#     # print(gbd.LBD)
#     lbds_proposed.append(gbd.LBD)
#     ubds_proposed.append(gbd.UBD)
#     while gbd.UBD - gbd.LBD > eps_:
#         # print("subproblem iteration")
#         gbd.solve_primal_problem(stage="Intermediate Step")
#         ubds_proposed.append(gbd.UBD)
#         if gbd.UBD - gbd.LBD > eps_:
#             # print("master problem iteration")
#             gbd.generate_graph_and_evaluate_agent()
#             gbd.evaluate_master_problem()
#             lbds_proposed.append(gbd.LBD)
#             pass 
#         pass 
    
#     if abs(gbd.LBD - gbd.UBD) <= eps_:
#         gbd.solve_primal_problem(stage="Final Step")
    
#     else:
#         gbd.set_new_lbds(gbd.LBD_cur)

#         while gbd.UBD - gbd.LBD > eps_:
#             # print("subproblem iteration")
#             gbd.solve_primal_problem(stage="Intermediate Step")
#             ubds_proposed.append(gbd.UBD)
#             if gbd.UBD - gbd.LBD > eps_:
#                 # print("master problem iteration")
#                 gbd.generate_graph_and_evaluate_agent()
#                 gbd.evaluate_master_problem()
#                 lbds_proposed.append(gbd.LBD)
#                 pass 

#             gbd.solve_primal_problem(stage="Final Step")


#     # gbd.solve_primal_problem(stage="Final Step")
#     print(gbd.UBD)
#     print(gbd.LBD_cur)
#     print(gbd.LBD)
#     return gbd.x, gbd.y, gbd.predicted_binary_values, gbd.predicted_values, gbd.LBD, lbds_proposed, ubds_proposed

# def gbd_algorithm(UBD, LBD, feas_pars, x_guess, y_guess, agent_type, eps_,
#                   gbd_env=None, lbds_proposed=None, ubds_proposed=None):
#     # Initialize the environment and tracking lists only once
#     if gbd_env is None:
#         gbd_env = GBDENV(UBD, LBD, feas_pars, x_guess, y_guess, eps_, agent_type)
#         lbds_proposed = [gbd_env.LBD]
#         ubds_proposed = [gbd_env.UBD]
#     else:
#         lbds_proposed.append(gbd_env.LBD)
#         ubds_proposed.append(gbd_env.UBD)

#     while gbd_env.UBD - gbd_env.LBD > eps_:
#         gbd_env.solve_primal_problem(stage="Intermediate Step")
#         ubds_proposed.append(gbd_env.UBD)

#         if gbd_env.UBD - gbd_env.LBD > eps_:
#             gbd_env.generate_graph_and_evaluate_agent()
#             gbd_env.evaluate_master_problem()
#             lbds_proposed.append(gbd_env.LBD)

#             if gbd_env.LBD > gbd_env.UBD:
#                 print("Restart triggered: LBD > UBD. Resetting LBD...")
#                 gbd_env.set_new_lbds(gbd_env.LBD_cur)  # Force LBD to a safe value
#                 return gbd_algorithm(
#                     gbd_env.UBD,
#                     gbd_env.LBD,
#                     feas_pars,
#                     gbd_env.x,
#                     gbd_env.y,
#                     agent_type,
#                     eps_,
#                     gbd_env,
#                     lbds_proposed,
#                     ubds_proposed
#                 )

#     # Final subproblem solve
#     gbd_env.solve_primal_problem(stage="Final Step")

#     print("Final UBD:", gbd_env.UBD)
#     print("Final LBD:", gbd_env.LBD)

#     return (
#         gbd_env.x,
#         gbd_env.y,
#         gbd_env.predicted_binary_values,
#         gbd_env.predicted_values,
#         gbd_env.LBD,
#         lbds_proposed,
#         ubds_proposed
#     )


def solve_problem_with_gbd_proposed(initial_guess, feasible_parameters, agent_type):
    UBD =  100000
    LBD = -100000
    x_guess = [1,1,1,1,1,1]
    eps_ = 0.001
    x_opt, y_opt, predicted_binary_values, predicted_values, optimal_value, lbds_proposed, ubds_proposed, time_mp = gbd_algorithm(UBD,LBD,feasible_parameters,x_guess,initial_guess,agent_type,eps_)
    # x_opt, y_opt, predicted_binary_values, predicted_values, optimal_value, lbds_proposed, ubds_proposed, time_mp = gbd_algorithm_new(UBD,LBD,feasible_parameters,x_guess,initial_guess,agent_type,eps_)
    return x_opt, y_opt, predicted_binary_values, predicted_values, optimal_value, lbds_proposed, ubds_proposed, time_mp
