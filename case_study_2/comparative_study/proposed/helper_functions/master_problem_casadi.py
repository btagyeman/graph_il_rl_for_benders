import casadi as cs
import numpy as np 
import contextlib
import os
import sys

# Context manager to suppress stdout and stderr
@contextlib.contextmanager
def suppress_output():
    with open(os.devnull, 'w') as devnull:
        with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
            yield
   
def master_problem_full_assignment(self):
    mus = []
    if len(self.dual_vars_opt)> 0:
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
                lagrange_opt+=self.R_c*self.c[j]
                lagrange_opt+=self.R_u*u_opt_cur[j]
                #terms in the constraints
                lagrange_opt+=dual_vars_opt_cur[idx_u_ubd[j]]*(u_opt_cur[j]-(self.c[j]*self.UB_mz1))
                lagrange_opt+=dual_vars_opt_cur[idx_u_lbd[j]]*((self.c[j]*self.UL_mz1)- u_opt_cur[j])
                lagrange_opt+=dual_vars_opt_cur[idx_x_ubd[j]]*(x_opt_cur[j]-self.UZ_scaled_mz1-eps_ubd_opt_cur[j])
                lagrange_opt+=dual_vars_opt_cur[idx_x_lbd[j]]*(self.LZ_scaled_mz1-eps_lbd_opt_cur[j]-x_opt_cur[j])
                pass 

            mus.append(lagrange_opt)
            pass 
        pass 

    mus = np.array(mus)

    self.lbd_cur = mus.max()
    lbd_temp = max(self.lbd_cur, self.lbd_prev)

    if lbd_temp > self.ubd:
        vanilla_master_problem(self)
        # self.c = [-1]*self.prediction_horizon
        # master_problem_partial_assignment(self)
    else:
        self.predicted_vars.extend(self.c)
        self.c = self.c
        self.mu_guess = mus.max() 
        self.c_guess = self.c 
        self.lbd = lbd_temp
        self.lbd_prev = self.lbd
        self.past_values = self.c
    return 


def master_problem_partial_assignment(self):
    w = []
    lbw = []
    ubw = []
    Guess = []
    discrete = []
    G   = []
    lbg = []
    ubg = []
    for i in range(1, self.prediction_horizon+1):
        c_current  = self.c[i-1]
        if c_current == -1:
            c_name = 'c' + str(i)
            Ck = cs.MX.sym(c_name, 1)
            w += [Ck]
            discrete += [True]
            lbw += [0]
            ubw += [1]
            Guess += [self.c_guess[i-1]]
        else:
            c_name = 'c' + str(i)
            Ck = cs.MX.sym(c_name, 1)
            w += [Ck]
            discrete += [False]
            lbw += [self.c[i-1]]
            ubw += [self.c[i-1]]
            Guess += [self.c[i-1]]
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
            pass 


    # solve the optimization problem using BONMIN
    milp = dict(f=J, g=cs.vertcat(*G), x=cs.vertcat(*w))
    opts ={}
    opts['discrete'] = discrete
    with suppress_output():
        solver = cs.nlpsol('solver', 'bonmin', milp, opts)
        r = solver(lbx=lbw, ubx=ubw, x0=Guess, lbg=lbg, ubg=ubg)
        pass 

    self.lbd_cur = r['f'].full().ravel()[0]
    lbd_temp = max(self.lbd_cur, self.lbd_prev)
    if lbd_temp < self.ubd:
        self.predicted_vars.extend(self.c) # add this before updating c
        opt_vals = r['x'].full().ravel()
        self.c = opt_vals[0:self.prediction_horizon].tolist()
        self.mu_guess = opt_vals[-1] # update the guess for the next iteration
        self.c_guess = opt_vals[0:self.prediction_horizon].tolist() # update the guess for the next iteration
        self.lbd = lbd_temp
        self.lbd_prev = self.lbd
        self.past_values = opt_vals[0:self.prediction_horizon].tolist()
    else:
        vanilla_master_problem(self)
    return 


def vanilla_master_problem(self):
    self.predicted_vars.extend(self.c)
    w = []
    lbw = []
    ubw = []
    Guess = []
    discrete = []
    G   = []
    lbg = []
    ubg = []
    for i in range(1, self.prediction_horizon+1):
        c_name = 'c' + str(i)
        Ck = cs.MX.sym(c_name, 1)
        w += [Ck]
        discrete += [True]
        lbw += [0]
        ubw += [1]
        Guess += [self.c_guess[i-1]]  # Use the guess for c
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
            pass 


    # solve the optimization problem using BONMIN
    milp = dict(f=J, g=cs.vertcat(*G), x=cs.vertcat(*w))
    opts ={}
    opts['discrete'] = discrete
    with suppress_output():
        solver = cs.nlpsol('solver', 'bonmin', milp, opts)
        r = solver(lbx=lbw, ubx=ubw, x0=Guess, lbg=lbg, ubg=ubg)
        pass 
    
    opt_vals = r['x'].full().ravel()
    self.c = opt_vals[0:self.prediction_horizon].tolist()
    self.mu_guess = opt_vals[-1] # update the guess for the next iteration
    self.c_guess = opt_vals[0:self.prediction_horizon].tolist() # update the guess for the next iteration
    self.lbd_cur = r['f'].full().ravel()[0]
    self.lbd = max(self.lbd_cur, self.lbd_prev)  # ensure lbd is non-decreasing
    self.lbd_prev = self.lbd
    self.past_values = opt_vals[0:self.prediction_horizon].tolist()
    return 
