from pyomo.environ import * 
import numpy as np

def master_problem_pyomo(self):
    model = ConcreteModel()
    model.bin_idx = RangeSet(self.prediction_horizon)
    model.c = Var(model.bin_idx, within=Binary, initialize=0, bounds=(0,1))
    model.mu = Var(within=Reals, initialize=0, bounds=(-10e19, 10e19))


    # Add the cuts if they exist
    if len(self.dual_vars_opt)> 0:
        print("Add the optimality cuts")
        for i in range(len(self.dual_vars_opt)):
            print(len(self.dual_vars_opt))
            u_opt_cur = self.opt_u[i]
            x_opt_cur = self.opt_x_sc[i]
            eps_lbd_opt_cur = self.opt_eps_lbd[i]
            eps_ubd_opt_cur = self.opt_eps_ubd[i]
            lagrange_terms_opt = 0 
            dual_vars_opt_cur = self.dual_vars_opt[i]

            idx_u_ubd = np.arange(1,(self.prediction_horizon*5),5).tolist()
            idx_u_lbd = np.arange(2,(self.prediction_horizon*5),5).tolist()
            idx_x_ubd = np.arange(3,(self.prediction_horizon*5),5).tolist()
            idx_x_lbd = np.arange(4,(self.prediction_horizon*5),5).tolist()  
            # for j in range(self.prediction_horizon):
            #     #terms in the cost function
            #     term_1 = self.QUpper_mz1*(eps_ubd_opt_cur[j]**2)
            #     term_2 = self.QLower_mz1*(eps_lbd_opt_cur[j]**2)
            #     term_3 = 1000*model.c[j+1] 
            #     term_4 = -9000*u_opt_cur[j]

            #     #terms in the constraints 
            #     term_5 = dual_vars_opt_cur[idx_u_ubd[j]]*(u_opt_cur[j] -(model.c[j+1]*self.UB_mz1))
            #     term_6 = dual_vars_opt_cur[idx_u_lbd[j]]*((model.c[j+1]*self.UL_mz1) - u_opt_cur[j])
            #     term_7 = dual_vars_opt_cur[idx_x_ubd[j]]*(x_opt_cur[j] - self.UZ_scaled_mz1 - eps_ubd_opt_cur[j])
            #     term_8 = dual_vars_opt_cur[idx_x_lbd[j]]*(self.LZ_scaled_mz1 - eps_lbd_opt_cur[j] - x_opt_cur[j])

            #     lagrange_terms_opt = term_1 + term_2 + term_3 + term_4 + term_5 + term_6 + term_7 + term_8
            #     pass       

            for j in model.bin_idx:
                #terms in the cost function
                term_1 = self.QUpper_mz1*(eps_ubd_opt_cur[j-1]**2)
                term_2 = self.QLower_mz1*(eps_lbd_opt_cur[j-1]**2)
                term_3 = 1000*model.c[j] 
                term_4 = -9000*u_opt_cur[j-1]

                #terms in the constraints 
                term_5 = dual_vars_opt_cur[idx_u_ubd[j-1]]*(u_opt_cur[j-1] -(model.c[j]*self.UB_mz1))
                term_6 = dual_vars_opt_cur[idx_u_lbd[j-1]]*((model.c[j]*self.UL_mz1) - u_opt_cur[j-1])
                term_7 = dual_vars_opt_cur[idx_x_ubd[j-1]]*(x_opt_cur[j-1] - self.UZ_scaled_mz1 - eps_ubd_opt_cur[j-1])
                term_8 = dual_vars_opt_cur[idx_x_lbd[j-1]]*(self.LZ_scaled_mz1 - eps_lbd_opt_cur[j-1] - x_opt_cur[j-1])

                lagrange_terms_opt = term_1 + term_2 + term_3 + term_4 + term_5 + term_6 + term_7 + term_8
                pass                        

            model.add_component('con_opt_' + str(i), Constraint(expr=model.mu>=lagrange_terms_opt))


                
    if len(self.dual_vars_feas)>0:
        print("Add the feasibility cuts")
        for k in range(len(self.dual_vars_feas)):
            u_feas_cur = self.feas_u[k].tolist()
            x_feas_cur = self.feas_x_sc[k].tolist()
            eps_lbd_feas_cur = self.feas_eps_lbd[k].tolist()
            eps_ubd_feas_cur = self.feas_eps_ubd[k].tolist()
            lagrange_terms_feas = 0 
            dual_vars_feas_cur = self.dual_vars_feas[k].tolist()

            idx_u_ubd = np.arange(1,(self.prediction_horizon*5),5).tolist()
            idx_u_lbd = np.arange(2,(self.prediction_horizon*5),5).tolist()
            idx_x_ubd = np.arange(3,(self.prediction_horizon*5),5).tolist()
            idx_x_lbd = np.arange(4,(self.prediction_horizon*5),5).tolist()

            for l in range(self.prediction_horizon):
                #terms in the constraints 
                term_1 = dual_vars_feas_cur[idx_u_ubd[l]]*(u_feas_cur[l] - (model.c[l+1]*self.UB_mz1))
                term_2 = dual_vars_feas_cur[idx_u_lbd[l]]*((model.c[l+1]*self.UL_mz1) - u_feas_cur[l])
                term_3 = dual_vars_feas_cur[idx_x_ubd[l]]*(x_feas_cur[j] - self.UZ_scaled_mz1 - eps_ubd_feas_cur[l])
                term_4 = dual_vars_feas_cur[idx_x_lbd[l]]*(self.LZ_scaled_mz1 - eps_lbd_feas_cur[l] - x_feas_cur[l])

                lagrange_terms_feas = term_1 + term_2 + term_3 + term_4
                pass 

            model.add_component('con_feas_' + str(k), Constraint(expr= 0>=lagrange_terms_feas))

    # add the objective function
    model.obj = Objective(expr=model.mu, sense=minimize)

    # solve the problem
    # solver = SolverFactory('knitroampl')
    solver = SolverFactory('gurobi')
    results = solver.solve(model)
    print(results.solver.termination_condition)
    #update the lower bound
    self.LBD = value(model.mu)
    opt_c = []

    for i in model.bin_idx:
        opt_c.append(value(model.c[1]))
        pass
    print(opt_c)
    self.c = opt_c
    self.c_guess = opt_c
    return 