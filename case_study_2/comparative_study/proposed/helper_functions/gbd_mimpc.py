import numpy as np 
from helper_functions.master_problem_casadi import master_problem_full_assignment, master_problem_partial_assignment, vanilla_master_problem
from helper_functions.graph_representation import create_graph_representation
from helper_functions.original_minlp import original_milp
from helper_functions.subproblem import subproblem
from spektral.data import Dataset
from spektral.data.loaders import DisjointLoader
import time
import tensorflow as tf 
from spektral.layers import GlobalSumPool, ECCConv
from tensorflow.keras.layers import Dense

x_mean = np.load('./trained_agent/x_mean.npy')
x_std = np.load('./trained_agent/x_std.npy')
e_mean = np.load('./trained_agent/e_mean.npy')
e_std = np.load('./trained_agent/e_std.npy')
epsilon = 1e-08

class ActorNetwork(tf.keras.Model):
    def __init__(self, num_outputs=14):
        super().__init__()
        self.conv1 = ECCConv(64, activation='relu')
        self.conv2 = ECCConv(64, activation='relu')
        self.pool = GlobalSumPool()
        self.dense1 = Dense(64, activation='relu')
        self.output_heads = [Dense(1, activation='sigmoid') for _ in range(num_outputs)]

    def call(self, inputs):
        x, a, e, i = inputs
        # # Normalize x
        x = (x - x_mean) / (x_std + epsilon)
        e = (e - e_mean) /(e_std + epsilon)
        x = self.conv1([x, a, e])
        x = self.conv2([x, a, e])
        x = self.pool([x, i])
        x = self.dense1(x)
        outputs = [head(x) for head in self.output_heads]  
        return tf.concat(outputs, axis=-1)
    
class GraphListDataset(Dataset):
    def __init__(self, graph_list, **kwargs):
        self.graph_list = graph_list
        super().__init__(**kwargs)
    def read(self):
        return self.graph_list
    
agent = ActorNetwork()
agent.load_weights('./trained_agent/variables/variables').expect_partial()

class GBD_MIMPC:    
    def __init__(self,currentStates,previousInputs,guessesX,guessesU,guessesC,cropCoeff,refEvap,rooting_depths,lai_factors,rain,ubd,lbd):      
        self.prediction_horizon = 14
        self.sequenceLength = 5 
        self.data_min_mz1 = np.loadtxt('../../trained_lstm_models/model_weights/data_min_mz1_vrd_lai.txt')
        self.data_max_mz1 = np.loadtxt('../../trained_lstm_models/model_weights/data_max_mz1_vrd_lai.txt')
        self.LowerZone_mz1 = 0.200
        self.UpperZone_mz1 = 0.280
        self.LZ_scaled_mz1 = (self.LowerZone_mz1-self.data_min_mz1[0])/(self.data_max_mz1[0]-self.data_min_mz1[0])
        self.UZ_scaled_mz1 = (self.UpperZone_mz1-self.data_min_mz1[0])/(self.data_max_mz1[0]-self.data_min_mz1[0])
        self.UL_mz1 = -0.0318400
        self.UB_mz1 = -0.0038880
        self.UL_mz1_mod = -0.0318400
        self.UB_mz1_mod = -0.0038880
        self.QUpper_mz1 = 9.2e03
        self.QLower_mz1 = 9.0e03
        self.R_c = 5 
        self.R_u = -50
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
        self.dual_vars_opt = []
        self.dual_vars_feas = []
        self.c_agent = guessesC
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
        self.ubd_prev = ubd 
        self.ubd_cur = ubd 
        self.ubd = ubd
        self.lbd = lbd
        self.lbd_cur = lbd 
        self.lbd_prev = lbd
        self.mu_guess = 0
        self.cut_order = []
        self.n_variables = self.prediction_horizon
        self.optimal_values = []
        self.predicted_vars = []
        self.past_values = guessesC
        self.optimal_values.append(self.c)  
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
        
    def obtain_irrig_amount(self, opt_values, seed_point):
        pres_irrig = []
        seed_point_u = 13
        for j in range(self.prediction_horizon):
            pres_irrig.append(opt_values[int(seed_point_u+(j*(seed_point)))])
            pass
        return pres_irrig
      
    def generate_graph_and_evaluate_agent(self):
        if len(self.dual_vars_opt)> 0:
            print("Add the optimality cuts")
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
            print("Add the feasibility cuts")
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
    

    def evaluate_master_problem(self, observation_graph=None):
        if observation_graph is not None:
            print("Evaluate the agent")
            with tf.device('/CPU:0'):
                y_pred = agent(observation_graph, training=False).numpy().squeeze()
                pass 
            THRESH_HIGH = 0.90
            THRESH_LOW = 0.10
            binary_pred = np.where(y_pred >= THRESH_HIGH, 1, np.where(y_pred <= THRESH_LOW, 0, -1))
            self.c = binary_pred.tolist()
            self.c_agent = binary_pred.tolist()
            pass 

        else:
            self.free_variables()

        comp_vars = np.array(self.c)

        if np.all(comp_vars == -1):
            vanilla_master_problem(self)
        
        elif np.any(comp_vars == -1):
            print("Partial assignment")
            master_problem_partial_assignment(self)
        else:
            print("Full assignment")
            master_problem_full_assignment(self)
        pass
    
    def solve_original_problem(self):
        return original_milp(self)
    
    def solve_subproblem(self,stage):
        return subproblem(self,stage)
    
    def free_variables(self):
        self.c = [-1]*self.prediction_horizon

    def set_new_lbds(self, new_lbd_candidate, epsilon=1e-4):
        self.lbd = min(self.ubd - epsilon, max(self.lbd_prev, new_lbd_candidate))
        self.lbd_prev = self.lbd
        


def run_gbd_for_scheduler(x0_scaled_mz1, u_init_mz1,guessesX,guessesU,guessesC,kc_scaled_mz1, et_scaled_mz1,rd_scaled_mzs,lai_scaled_mzs,rain,ubd,lbd,epsilon):
    gbd_instance = GBD_MIMPC(x0_scaled_mz1,u_init_mz1,guessesX,guessesU.tolist(),guessesC.tolist(),kc_scaled_mz1,et_scaled_mz1,rd_scaled_mzs,lai_scaled_mzs,rain,ubd,lbd)
    eval_time_mp = []
    mp_count = 0
    bin_var_predicted = []
    converged = False
    restart_attempts = 0
    max_restarts = 10

    while not converged and restart_attempts <= max_restarts:
        while gbd_instance.ubd - gbd_instance.lbd > epsilon:
            gbd_instance.solve_subproblem(stage='Intermediate Step')
            observation_graph = gbd_instance.generate_graph_and_evaluate_agent()
        
            print("Current bounds (subproblem): ubd = ", gbd_instance.ubd)
            if gbd_instance.ubd - gbd_instance.lbd > epsilon:
                t_init = time.time()
                gbd_instance.evaluate_master_problem(observation_graph)
                mp_count += 1
                bin_var_predicted.append(gbd_instance.c_agent)
                t_final = time.time()
                eval_time_mp.append(t_final - t_init)   
                print("Current bounds (masterproblem): lbd = ", gbd_instance.lbd)
                pass 
            pass

        if abs(gbd_instance.ubd - gbd_instance.lbd)<= 0.7: 
            print("Final bounds: lbd = ", gbd_instance.lbd, " and ubd = ", gbd_instance.ubd)
            print("Final irrigation decisions", gbd_instance.c)
            gbd_instance.solve_subproblem(stage='Final Step')
            converged = True


        elif gbd_instance.lbd > gbd_instance.ubd + 0.7:
            gbd_instance.set_new_lbds(gbd_instance.lbd_cur)
            # observation_graph = gbd_instance.generate_graph_and_evaluate_agent()
            restart_attempts += 1
            # gbd_instance.free_variables()
            # gbd_instance.evaluate_master_problem(observation_graph)
            gbd_instance.evaluate_master_problem(observation_graph=None)

            while gbd_instance.ubd - gbd_instance.lbd > epsilon:
                gbd_instance.solve_subproblem(stage='Intermediate Step')
                observation_graph = gbd_instance.generate_graph_and_evaluate_agent()
                print("Current bounds (subproblem): ubd = ", gbd_instance.ubd)
                if gbd_instance.ubd - gbd_instance.lbd > epsilon:
                    t_init = time.time()
                    gbd_instance.evaluate_master_problem(observation_graph)
                    mp_count += 1
                    bin_var_predicted.append(gbd_instance.c_agent)
                    t_final = time.time()
                    eval_time_mp.append(t_final - t_init)   
                    print("Current bounds (masterproblem): lbd = ", gbd_instance.lbd)
                    pass 
                pass

            if abs(gbd_instance.ubd - gbd_instance.lbd)<= 0.7: 
                print("Final bounds: lbd = ", gbd_instance.lbd, " and ubd = ", gbd_instance.ubd)
                print("Final irrigation decisions", gbd_instance.c)
                gbd_instance.solve_subproblem(stage='Final Step')
                converged = True
                pass 
            pass 
    if not converged:
        print("[Error] GBD failed to converge after max restarts. Final LBD still exceeds UBD.")

    return gbd_instance.x,gbd_instance.u,gbd_instance.c,gbd_instance.ubd,gbd_instance.lbd,gbd_instance.optimal_values,gbd_instance.graph_features,gbd_instance.cut_order,eval_time_mp, bin_var_predicted, mp_count,gbd_instance.predicted_vars
