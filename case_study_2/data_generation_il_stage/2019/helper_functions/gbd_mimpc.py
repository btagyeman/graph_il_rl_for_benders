import numpy as np 
from helper_functions.master_problem_casadi import master_problem_casadi_sub_opt, master_problem_casadi_opt
from helper_functions.master_problem_pyomo import master_problem_pyomo
from helper_functions.original_minlp import original_milp
from helper_functions.subproblem import subproblem

class GBD_MIMPC:    
    def __init__(self,currentStates,previousInputs,guessesX,guessesU,guessesC,cropCoeff,refEvap,rooting_depths,lai_factors,rain,ubd,lbd):      
        self.prediction_horizon = 14
        # self.prediction_horizon = 7
        self.sequenceLength = 5 # this is fixed for now
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
        # self.factor = 10000
        self.factor = 1
        # self.QUpper_mz1 = self.factor*1.2e06
        # self.QLower_mz1 = self.factor*1.0e06

        self.QUpper_mz1 = self.factor*9.2e03
        self.QLower_mz1 = self.factor*9.0e03
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
        self.mu_guess = 0
        #for data collection purposes
        self.cut_order = []
        self.optimal_values = [] # Store the optimal values of the complicating variables after each iteration of the master problem
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
        # do not scale the slack varaibles for now
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
    
    def solve_original_problem(self):
        return original_milp(self)
    
    def solve_master_problem_casadi_sub_opt(self):
        #Optimal values of the  non complicating variables are used to generate the cuts, together with the dual variables.
        return master_problem_casadi_sub_opt(self)
    
    def solve_master_problem_casadi_opt(self):
        #An unconstrained problem involving the Lagrangian and the dual variables is solved
        return master_problem_casadi_opt(self)
    
    def solve_subproblem(self,stage):
        return subproblem(self,stage)

def run_gbd_for_scheduler(x0_scaled_mz1, u_init_mz1,guessesX,guessesU,guessesC,
    kc_scaled_mz1, et_scaled_mz1,rd_scaled_mzs,lai_scaled_mzs,rain,ubd,lbd,epsilon):
    
    gbd_instance = GBD_MIMPC(x0_scaled_mz1,u_init_mz1,guessesX,guessesU.tolist(),guessesC.tolist(),
            kc_scaled_mz1,et_scaled_mz1,rd_scaled_mzs,lai_scaled_mzs,rain,ubd,lbd)
    
    last_master_solved = False
    
    while gbd_instance.ubd - gbd_instance.lbd > epsilon:
        gbd_instance.solve_subproblem(stage='Intermediate Step')
        print("Current bounds (subproblem): ubd = ", gbd_instance.ubd,  "lbd = ", gbd_instance.lbd)
        if gbd_instance.ubd - gbd_instance.lbd > epsilon:
            gbd_instance.solve_master_problem_casadi_sub_opt()
            print("Current bounds (masterproblem): ubd = ", gbd_instance.ubd,  "lbd = ", gbd_instance.lbd)
            last_master_solved = True
            pass
        else:
            last_master_solved = False
        pass 
    # solve the subproblem one last time to get the final values
    gbd_instance.solve_subproblem(stage='Final Step')
    print("Final irrigation decisions", gbd_instance.c)
    if not last_master_solved:
        gbd_instance.solve_master_problem_casadi_sub_opt()
        pass
    return gbd_instance.x,gbd_instance.u,gbd_instance.c,gbd_instance.ubd,gbd_instance.lbd,gbd_instance.optimal_values,gbd_instance.graph_features,gbd_instance.cut_order
