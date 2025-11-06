import numpy as np 
from helper_functions.master_problems import master_problem_rl, master_problem_il
from helper_functions.primal_problems import initial_subproblem, subproblem
from helper_functions.graph_representation import create_graph_representation
from common.actual_field.temporal_spatial_parameters_field import *
from common.actual_field.model_simulation import simulate_Richards_equation
from common.utils_season_long import compute_root_zone_moisture,obtain_soil_moisture
from common.actual_field.richards_equation_field import volMoistureAllNodes as vwc_fun

from spektral.data import Dataset
from spektral.data.loaders import DisjointLoader
soilPars_mz1 = pars_mz1()

class GraphListDataset(Dataset):
    def __init__(self, graph_list, **kwargs):
        self.graph_list = graph_list
        super().__init__(**kwargs)
    def read(self):
        return self.graph_list
    
class GBD_MIMPC:    
    def __init__(self,currentStates,previousInputs,guessesX,guessesU,guessesC,cropCoeff,refEvap,rooting_depths,lai_factors,rain,ubd,lbd, epsilon):      
        self.prediction_horizon = 14
        self.sequenceLength = 5 
        self.data_min_mz1 = np.loadtxt('./trained_lstm_models/model_weights/data_min_mz1_vrd_lai.txt')
        self.data_max_mz1 = np.loadtxt('./trained_lstm_models/model_weights/data_max_mz1_vrd_lai.txt')
        self.LowerZone_mz1 = 0.200
        self.UpperZone_mz1 = 0.280
        self.LZ_scaled_mz1 = (self.LowerZone_mz1-self.data_min_mz1[0])/(self.data_max_mz1[0]-self.data_min_mz1[0])
        self.UZ_scaled_mz1 = (self.UpperZone_mz1-self.data_min_mz1[0])/(self.data_max_mz1[0]-self.data_min_mz1[0])
        self.UL_mz1 = -0.0318400
        self.UB_mz1 = -0.0038880
        self.QUpper_mz1 = 9.2e03
        self.QLower_mz1 = 9.0e03
        self.R_c = 5 
        self.R_u = -50
        self.epsilon = epsilon
        self.max_iter = 10000

        self.ubd_prev = ubd 
        self.ubd_cur = ubd 
        self.ubd = ubd
        self.lbd = lbd

        self.ubd_cur = ubd 
        self.ubd = ubd

        # reward
        self.reward_infs_mp = 0
        self.reward_il = 0
        self.reward_time = 0
        self.reward_bnds = 0
        self.previous_gap = self.ubd - self.lbd
        self.initial_gap = self.ubd - self.lbd
        self.mp_feasibility = None

        self.currentStates = currentStates
        self.previousInputs = previousInputs
        self.guessesX = guessesX
        self.guessesU = guessesU
        self.guessesC = guessesC
        self.cropCoeff = cropCoeff
        self.refEvap = refEvap
        self.rooting_depths = rooting_depths
        self.lai_factors = lai_factors
        self.rain = rain
        x = np.array(guessesX)*(self.data_max_mz1[0]-self.data_min_mz1[0]) + self.data_min_mz1[0]
        self.x = x.tolist()
        self.x_guess = guessesX
        self.u = guessesU
        self.u_guess = guessesU
        self.eps_lbd = [0]*self.prediction_horizon 
        self.e_lbd_guess = [0]*self.prediction_horizon
        self.eps_ubd = [0]*self.prediction_horizon
        self.e_lbd_guess = [0]*self.prediction_horizon
        self.c = guessesC
        self.c_guess = guessesC

        self.modified_action = self.c

        self.c_il = guessesC
        self.c_guess_il = guessesC

        # for the gbd and graph construction
        self.dual_vars_opt = []
        self.dual_vars_feas = []
        self.opt_u = []
        self.opt_x = []
        self.opt_x_sc = []
        self.opt_eps_lbd = []
        self.opt_eps_ubd = []
        self.feas_u = []
        self.feas_x = []
        self.feas_x_sc = []
        self.feas_eps_lbd = []
        self.feas_eps_ubd = []

        self.mu_guess = 0

        self.cut_order = []
        self.past_values = guessesC
        self.n_variables = self.prediction_horizon
        self.graph_features = {}
        self.graph_features['optimality_cuts']={}
        self.graph_features['feasibility_cuts']= {}
        return 
        
    def scale_irrig_amount(self, u):
        u = u/86400     
        return (u-self.data_min_mz1[1])/(self.data_max_mz1[1]-self.data_min_mz1[1])
        
    def obtain_soil_moisture(self, opt_values, seed_point):
        pred_traj = []
        seed_point_x = 15
        for j in range(self.prediction_horizon):
            pred_traj.append(opt_values[int(seed_point_x+(j*(seed_point)))])
            pass
        pred_traj_unsc= (np.array(pred_traj)*(self.data_max_mz1[0]-self.data_min_mz1[0])) + self.data_min_mz1[0]
        return pred_traj, pred_traj_unsc.tolist()        

    def obtain_slack_vars(self, opt_values, seed_point):
        pred_eps_lbd = []
        pred_eps_ubd = []
        seed_point_eps_lbd=16
        seed_point_eps_ubd=17
        for j in range(self.prediction_horizon):
            pred_eps_lbd.append(opt_values[int(seed_point_eps_lbd+(j*(seed_point)))])
            pred_eps_ubd.append(opt_values[int(seed_point_eps_ubd+(j*(seed_point)))])
            pass
        return pred_eps_lbd, pred_eps_ubd
    
    def generate_graph_and_evaluate_agent(self):
        if len(self.dual_vars_opt)> 0:
            # print("Add the optimality cuts")
            for i in range(len(self.dual_vars_opt)):
                u_opt_cur = self.opt_u[i]
                x_opt_cur = self.opt_x_sc[i]
                eps_lbd_opt_cur = self.opt_eps_lbd[i]
                eps_ubd_opt_cur = self.opt_eps_ubd[i]
                dual_vars_opt_cur = self.dual_vars_opt[i]
                idx_u_ubd = np.arange(1,(self.prediction_horizon*5),5).tolist()
                idx_u_lbd = np.arange(2,(self.prediction_horizon*5),5).tolist()
                idx_x_ubd = np.arange(3,(self.prediction_horizon*5),5).tolist()
                idx_x_lbd = np.arange(4,(self.prediction_horizon*5),5).tolist()
                coeffs_opt_cut=[]
                constant_term_cut=0
                for j in range(self.prediction_horizon):

                    coeff = -(dual_vars_opt_cur[idx_u_ubd[j]]*self.UB_mz1) + (dual_vars_opt_cur[idx_u_lbd[j]]*self.UL_mz1) + self.R_c

                    constant_term_iter=self.QUpper_mz1*(eps_ubd_opt_cur[j]) + self.QLower_mz1*(eps_lbd_opt_cur[j]) + \
                    (self.R_u*u_opt_cur[j]) + dual_vars_opt_cur[idx_u_ubd[j]]*u_opt_cur[j] + (-dual_vars_opt_cur[idx_u_lbd[j]]*u_opt_cur[j]) + \
                    dual_vars_opt_cur[idx_x_ubd[j]]*(x_opt_cur[j]-self.UZ_scaled_mz1-eps_ubd_opt_cur[j]) + \
                    dual_vars_opt_cur[idx_x_lbd[j]]*(self.LZ_scaled_mz1-eps_lbd_opt_cur[j]-x_opt_cur[j])

                    coeffs_opt_cut.append(coeff)

                    constant_term_cut+=constant_term_iter 
                    pass 
                coeffs_opt_cut.append(-constant_term_cut)
                self.graph_features['optimality_cuts'][i+1] = coeffs_opt_cut
                pass 
            pass 


        if len(self.dual_vars_feas)>0:
            # print("Add the feasibility cuts")
            for i in range(len(self.dual_vars_feas)):
                u_feas_cur = self.feas_u[i].tolist()
                x_feas_cur = self.feas_x_sc[i].tolist()
                eps_lbd_feas_cur = self.feas_eps_lbd[i].tolist()
                eps_ubd_feas_cur = self.feas_eps_ubd[i].tolist()
                dual_vars_feas_cur = self.dual_vars_feas[i].tolist()
                idx_u_ubd = np.arange(1,(self.prediction_horizon*5),5).tolist()
                idx_u_lbd = np.arange(2,(self.prediction_horizon*5),5).tolist()
                idx_x_ubd = np.arange(3,(self.prediction_horizon*5),5).tolist()
                idx_x_lbd = np.arange(4,(self.prediction_horizon*5),5).tolist()
                coeffs_feas_cut = []
                constant_term_feas_cut = 0
            
                for j in range(self.prediction_horizon):

                    coeff = -(dual_vars_feas_cur[idx_u_ubd[j]]*self.UB_mz1) + (dual_vars_feas_cur[idx_u_lbd[j]]*self.UL_mz1)

                    constant_term_iter = dual_vars_feas_cur[idx_x_ubd[j]]*(x_feas_cur[j]-self.UZ_scaled_mz1-eps_ubd_feas_cur[j]) + \
                                    dual_vars_feas_cur[idx_x_lbd[j]]*(self.LZ_scaled_mz1-eps_lbd_feas_cur[j]-x_feas_cur[j])+ \
                                    dual_vars_feas_cur[idx_u_ubd[j]]*u_feas_cur[j] + (-dual_vars_feas_cur[idx_u_lbd[j]]*u_feas_cur[j])
                    
                    constant_term_feas_cut += constant_term_iter
                    coeffs_feas_cut.append(coeff)
                    pass 
                coeffs_feas_cut.append(-constant_term_feas_cut)
                self.graph_features['feasibility_cuts'][i+1] = coeffs_feas_cut
                pass 
            pass 

        current_graph = create_graph_representation(self.cut_order, self.graph_features,self.past_values, self.n_variables)
        loader = DisjointLoader(GraphListDataset([current_graph]), batch_size=1, epochs=1)
        observation_graph, _ = next(iter(loader))
        return observation_graph 

        
    def obtain_irrig_amount(self, opt_values, seed_point):
        pres_irrig = []
        seed_point_u = 13
        for j in range(self.prediction_horizon):
            pres_irrig.append(opt_values[int(seed_point_u+(j*(seed_point)))])
            pass
        return pres_irrig
    
    
    def solve_master_problem_il(self):
        return master_problem_il(self)

    def solve_master_problem_rl(self, actions):
        return master_problem_rl(self, actions)
    
    def solve_initial_subproblem(self):
        return initial_subproblem(self)

    def solve_subproblem(self,stage):
        return subproblem(self,stage)

    def compute_rewards(self):
        if self.mp_feasibility == 'True':
            current_gap = self.ubd - self.lbd
            self.reward_bnds = (self.previous_gap - current_gap)/self.initial_gap 
            self.previous_gap = current_gap
            error_c = np.array(self.c_il) - np.array(self.c)
            error_c = np.abs(error_c)
            self.reward_il = np.sum(error_c)/ self.n_variables # Normalize the error to prevent it form exploding
            self.reward_infs_mp = 0.25*2
        else:
            self.reward_bnds = 0 
            self.reward_il = 0 
            self.reward_time = 0
   
  
        self.reward_time = -min(-self.reward_time, 1)
        total_reward = 6*self.reward_bnds - 3*self.reward_il + self.reward_infs_mp + 1*self.reward_time
        # total_reward = 6*self.reward_bnds + self.reward_infs_mp 
        return total_reward
    
    def step(self, actions):

        self.solve_master_problem_rl(actions)
        # self.solve_master_problem_rl_modified(actions)
        self.solve_master_problem_il() 
        self.solve_subproblem(stage='Intermediate Step')
        obs = self.generate_graph_and_evaluate_agent()

        reward = self.compute_rewards()
        modified_action = self.modified_action
        done = (self.ubd - self.lbd) <= self.epsilon or len(self.cut_order) >= self.max_iter
        info = {"lbd": self.lbd,"ubd": self.ubd,"feasible": self.mp_feasibility,"cuts": len(self.cut_order)}
        return obs, reward, done, info, modified_action
    
    def obtain_next_vwc(self, allHeadValues, et, rain, kc):
        self.solve_subproblem(stage='Final Step')
        irrig_amnt = self.u[0]
        rd = 0.5 
        lai = 1 
        total_irrig = (irrig_amnt + rain)/86400
        x_=simulate_Richards_equation(allHeadValues,total_irrig,et,kc,4,rd,lai,soilPars_mz1)
        vol_=obtain_soil_moisture(x_, soilPars_mz1,vwc_fun)
        rootzone_moist_=compute_root_zone_moisture(vol_, rd)
        return self.u, self.x, self.c, x_, vol_, rootzone_moist_

    def reset(self):
        self.solve_initial_subproblem()
        return self.generate_graph_and_evaluate_agent()
