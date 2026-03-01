import casadi as cs
import numpy as np 
from pyomo.environ import *
from helper_functions.minimize_lagrangian import optimize_lagrangian

def master_problem_casadi_sub_opt(self):
    w = []
    lbw = []
    ubw = []
    Guess = []
    discrete = []
    G   = []
    lbg = []
    ubg = []
    for i in range(1, self.prediction_horizon+1):
        c_name='c'+str(i)
        Ck=cs.MX.sym(c_name, 1)
        w+=[Ck]
        discrete+=[True]
        lbw+=[0]
        ubw+=[1]
        Guess+=[self.c_guess[i-1]]
        pass

    mu_name = 'mu'
    mu = cs.MX.sym(mu_name, 1)
    w += [mu]
    discrete += [False]
    lbw += [-cs.inf]
    ubw += [cs.inf]
    Guess += [self.mu_guess]
    J = mu
    
    # Add the cuts if they exist
    if len(self.dual_vars_opt)> 0:
        print("Add the optimality cuts")
        for i in range(len(self.dual_vars_opt)):
            u_opt_cur = self.opt_u[i]
            x_opt_cur = self.opt_x_sc[i]
            eps_lbd_opt_cur = self.opt_eps_lbd[i]
            eps_ubd_opt_cur = self.opt_eps_ubd[i]
            lagrange_opt=0 
            dual_vars_opt_cur = self.dual_vars_opt[i]
            idx_u_ubd = np.arange(1,(self.prediction_horizon*5),5).tolist()
            idx_u_lbd = np.arange(2,(self.prediction_horizon*5),5).tolist()
            idx_x_ubd = np.arange(3,(self.prediction_horizon*5),5).tolist()
            idx_x_lbd = np.arange(4,(self.prediction_horizon*5),5).tolist()

            for j in range(self.prediction_horizon):
                lagrange_opt+=self.QUpper_mz1*(eps_ubd_opt_cur[j])
                lagrange_opt+=self.QLower_mz1*(eps_lbd_opt_cur[j])
                lagrange_opt+=self.R_c*w[j]
                lagrange_opt+=self.R_u*u_opt_cur[j]
                #terms in the constraints
                lagrange_opt+=dual_vars_opt_cur[idx_u_ubd[j]]*(u_opt_cur[j]-cs.mtimes(w[j],self.UB_mz1))
                lagrange_opt+=dual_vars_opt_cur[idx_u_lbd[j]]*(cs.mtimes(w[j],self.UL_mz1)- u_opt_cur[j])
                lagrange_opt+=dual_vars_opt_cur[idx_x_ubd[j]]*(x_opt_cur[j]-self.UZ_scaled_mz1-eps_ubd_opt_cur[j])
                lagrange_opt+=dual_vars_opt_cur[idx_x_lbd[j]]*(self.LZ_scaled_mz1-eps_lbd_opt_cur[j]-x_opt_cur[j])
                pass 
            G+=[lagrange_opt-w[-1]]
            lbg+=[-cs.inf]
            ubg+=[0]
       
    if len(self.dual_vars_feas)>0:
        print("Add the feasibility cuts")
        for i in range(len(self.dual_vars_feas)):
            u_feas_cur = self.feas_u[i].tolist()
            x_feas_cur = self.feas_x_sc[i].tolist()
            eps_lbd_feas_cur = self.feas_eps_lbd[i].tolist()
            eps_ubd_feas_cur = self.feas_eps_ubd[i].tolist()
            lagrange_feas = 0 
            dual_vars_feas_cur = self.dual_vars_feas[i].tolist()
            idx_u_ubd = np.arange(1,(self.prediction_horizon*5),5).tolist()
            idx_u_lbd = np.arange(2,(self.prediction_horizon*5),5).tolist()
            idx_x_ubd = np.arange(3,(self.prediction_horizon*5),5).tolist()
            idx_x_lbd = np.arange(4,(self.prediction_horizon*5),5).tolist()

            
            for j in range(self.prediction_horizon):
                #terms in the constraints 
                lagrange_feas+=dual_vars_feas_cur[idx_u_ubd[j]]*(u_feas_cur[j]-cs.mtimes(w[j], self.UB_mz1))
                lagrange_feas+=dual_vars_feas_cur[idx_u_lbd[j]]*(cs.mtimes(w[j],self.UL_mz1)-u_feas_cur[j])
                lagrange_feas+=dual_vars_feas_cur[idx_x_ubd[j]]*(x_feas_cur[j]-self.UZ_scaled_mz1-eps_ubd_feas_cur[j])
                lagrange_feas+=dual_vars_feas_cur[idx_x_lbd[j]]*(self.LZ_scaled_mz1-eps_lbd_feas_cur[j] - x_feas_cur[j])
                pass 
            G+=[lagrange_feas]
            lbg+=[-cs.inf]
            ubg+=[0]

    # solve the optimization problem using BONMIN
    milp = dict(f=J, g=cs.vertcat(*G), x=cs.vertcat(*w))
    opts ={}
    opts['discrete'] = discrete
    opts['print_time'] = 0
    opts['bonmin.print_level'] = 0 # up to 12 level verbosity
    opts['bonmin.sb'] = "yes" # no ipopt banner
    opts['bonmin.bb_log_level'] = 0
    opts["bonmin.nlp_log_level"] = 0
    opts["bonmin.nlp_log_at_root"] = 0
    opts["bonmin.oa_log_level"] = 0
    opts["bonmin.lp_log_level"] = 0
    opts["bonmin.fp_log_level"] = 0
    opts["bonmin.milp_log_level"] = 0
    solver = cs.nlpsol('solver', 'bonmin', milp, opts)
    r = solver(lbx=lbw, ubx=ubw, x0=Guess, lbg=lbg, ubg=ubg)
    opt_vals = r['x'].full().ravel()
    self.c = opt_vals[0:self.prediction_horizon].tolist()
    self.optimal_values.append(self.c)  # Store the optimal values of the complicating variables after each iteration of the master problem
    self.mu_guess = opt_vals[-1] # update the guess for the next iteration
    self.c_guess = opt_vals[0:self.prediction_horizon].tolist() # update the guess for the next iteration
    self.lbd = r['f'].full().ravel()[0]
    return 

def master_problem_casadi_opt(self):
    w = []
    lbw = []
    ubw = []
    Guess = []
    discrete = []
    G   = []
    lbg = []
    ubg = []

    for i in range(1, self.prediction_horizon+1):
        c_name= 'c' + str(i)
        Ck=cs.MX.sym(c_name, 1)
        w+=[Ck]
        discrete+=[True]
        lbw+=[0]
        ubw+=[1]
        Guess+=[self.c_guess[i-1]]
        pass

    mu_name = 'mu'
    mu = cs.MX.sym(mu_name, 1)
    w += [mu]
    discrete += [False]
    lbw += [-cs.inf]
    ubw += [cs.inf]
    Guess += [self.mu_guess]
    J = mu

    # Add the cuts if they exist
    if len(self.dual_vars_opt)> 0:
        print("Add the optimality cuts")
        for i in range(len(self.dual_vars_opt)):
            u_opt_cur = self.opt_u[i]
            x_opt_cur = self.opt_x_sc[i]
            eps_lbd_opt_cur = self.opt_eps_lbd[i]
            eps_ubd_opt_cur = self.opt_eps_ubd[i]
            guesses = {}
            guesses['x'] = x_opt_cur
            guesses['u'] = u_opt_cur
            guesses['e_lbd'] = eps_lbd_opt_cur
            guesses['e_ubd'] = eps_ubd_opt_cur
            lagrange_opt = 0 
            dual_vars_opt_cur = self.dual_vars_opt[i]
            cost = optimize_lagrangian(self, dual_vars_opt_cur, guesses)
            idx_u_ubd=np.arange(1,(self.prediction_horizon*5),5).tolist()
            idx_u_lbd=np.arange(2,(self.prediction_horizon*5),5).tolist()
            idx_x_ubd=np.arange(3,(self.prediction_horizon*5),5).tolist()
            idx_x_lbd=np.arange(4,(self.prediction_horizon*5),5).tolist()  
            
            for j in range(self.prediction_horizon):
                #terms in the cost function
                # lagrange_opt+=1000*w[j]  # this comes from the cost function
                lagrange_opt+=5*w[j]  # this comes from the cost function
                lagrange_opt+=dual_vars_opt_cur[idx_u_ubd[j]]*(-cs.mtimes(w[j],self.UB_mz1)) # this comes from the constraint
                lagrange_opt+=dual_vars_opt_cur[idx_u_lbd[j]]*(cs.mtimes(w[j],self.UL_mz1)) # this comes from the constraint
                pass 
            lagrange_opt+=cost
            G+=[lagrange_opt-w[-1]]
            lbg+=[-cs.inf]
            ubg+=[0]

    #TO DO: Enhance this part of the code by directly minizing the Lagrangian over the non-complicating variables              
    if len(self.dual_vars_feas)>0:
        print("Add the feasibility cuts")
        for i in range(len(self.dual_vars_feas)):
            u_feas_cur = self.feas_u[i].tolist()
            x_feas_cur = self.feas_x_sc[i].tolist()
            eps_lbd_feas_cur = self.feas_eps_lbd[i].tolist()
            eps_ubd_feas_cur = self.feas_eps_ubd[i].tolist()
            lagrange_feas = 0 
            dual_vars_feas_cur = self.dual_vars_feas[i].tolist()
            idx_u_ubd = np.arange(1,(self.prediction_horizon*5),5).tolist()
            idx_u_lbd = np.arange(2,(self.prediction_horizon*5),5).tolist()
            idx_x_ubd = np.arange(3,(self.prediction_horizon*5),5).tolist()
            idx_x_lbd = np.arange(4,(self.prediction_horizon*5),5).tolist()

            for j in range(self.prediction_horizon):
                #terms in the constraints 
                lagrange_feas+=dual_vars_feas_cur[idx_u_ubd[j]]*(u_feas_cur[j] - cs.mtimes(w[j], self.UB_mz1))
                lagrange_feas+=dual_vars_feas_cur[idx_u_lbd[j]]*(cs.mtimes(w[j], self.UL_mz1) - u_feas_cur[j])
                lagrange_feas+=dual_vars_feas_cur[idx_x_ubd[j]]*(x_feas_cur[j] - self.UZ_scaled_mz1 - eps_ubd_feas_cur[j])
                lagrange_feas+=dual_vars_feas_cur[idx_x_lbd[j]]*(self.LZ_scaled_mz1 - eps_lbd_feas_cur[j] - x_feas_cur[j])
                pass 
            G+=[lagrange_feas]
            lbg+=[-cs.inf]
            ubg+=[0]

    # solve the optimization problem using BONMIN
    milp = dict(f=J, g=cs.vertcat(*G), x=cs.vertcat(*w))
    opts = {"discrete": discrete, "bonmin.print_level": 0}
    solver = cs.nlpsol('solver', 'bonmin', milp, opts) 
    r = solver(lbx=lbw, ubx=ubw, x0=Guess, lbg=lbg, ubg=ubg)
    opt_vals = r['x'].full().ravel()
    self.c = opt_vals[0:self.prediction_horizon].tolist()
    self.mu_guess = opt_vals[-1]
    self.lbd = r['f'].full().ravel()[0]
    return 