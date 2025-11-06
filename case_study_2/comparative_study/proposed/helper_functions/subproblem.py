import casadi as cs
from pyomo.environ import *
import numpy as np 
import sys
sys.path.append('../../trained_lstm_models')
from lstm_mz1 import optimizeInputModel_automated_mz1
      
def subproblem(self, stage='Intermediate Step'):
    w = []
    J = 0
    lbw = []
    ubw = []
    Guess = []
    G = []
    lbg = []
    ubg = []
    pastStateValues = []
    pastInputValues = []
    
    for j in range(self.sequenceLength):
        xname = 'X' + str(j)
        Xk = cs.MX.sym(xname, 1)
        w+=[Xk]
        lbw+=[self.currentStates[j]]
        ubw+=[self.currentStates[j]]
        Guess+= [self.currentStates[j]]
        pastStateValues.append(Xk)
        pass
        
    for k in range(self.sequenceLength - 1):
        uname = 'U' +str(k)
        Uk = cs.MX.sym(uname, 1)
        w+= [Uk]
        lbw+=[self.previousInputs[k]]
        ubw+=[self.previousInputs[k]]
        Guess+= [self.previousInputs[k]]
        cname = 'C' + str(k)
        ck = cs.MX.sym(cname, 1)
        w+= [ck]
        lbw+=[1]
        ubw+=[1]
        Guess+=[1]
        pastInputValues.append(Uk)
        pass 

    for k in range(1, self.prediction_horizon+1):
        Uname = 'u' + str(k-1)
        Uk = cs.MX.sym(Uname, 1)    
        w+= [Uk]
        lbw+=[-cs.inf]
        ubw+=[cs.inf]
        Guess+= [self.u_guess[k-1]]
        pastInputValues.append(Uk)
        cname = 'C' + str(k-1)
        Ck = cs.MX.sym(cname,1)
        w+=[Ck]
        lbw+=[self.c[k-1]]
        ubw+=[self.c[k-1]]
        Guess+=[self.c[k-1]]
        currentStateValues = pastStateValues[k-1 : (k-1) + self.sequenceLength]
        currentInputValues = pastInputValues[k-1 : (k-1) + self.sequenceLength]   
        currentCropCoefficient = self.cropCoeff[k-1 : (k-1) + self.sequenceLength]
        currentRefEvap = self.refEvap[k-1 : (k-1) + self.sequenceLength]
        currentRootingDepths = self.rooting_depths[k-1: (k-1) + self.sequenceLength]
        currentLAI_factors = self.lai_factors[k-1: (k-1) + self.sequenceLength]
        currentRainValues = self.rain[k-1 : (k-1) + self.sequenceLength] 

        currentPrecipValues_unscaled = []
        assert len(currentInputValues) == len(currentRainValues)
        for i in range(len(currentRainValues)):
            currentPrecipValues_unscaled.append(currentInputValues[i]+currentRainValues[i])
            pass 
        
        currentPrecipValues_scaled = []
        for i in range(len(currentRainValues)):
            currentPrecipValues_scaled.append(self.scale_irrig_amount((currentPrecipValues_unscaled[i])))
            pass 

        Fk = optimizeInputModel_automated_mz1(currentStateValues,currentPrecipValues_scaled,currentCropCoefficient,currentRefEvap,currentRootingDepths,currentLAI_factors)

        Xname = 'X' +str(k)
        Xk=cs.MX.sym(Xname, 1)
        w+=[Xk]
        lbw+=[-cs.inf]
        ubw+=[cs.inf]
        Guess+=[self.x_guess[k-1]]
        pastStateValues.append(Xk)
        
        elName = 'el' + str(k)
        ekL = cs.MX.sym(elName, 1)  
        w+=[ekL]
        lbw+=[0]
        ubw+=[cs.inf]
        Guess+=[0]
        
        euName = 'el'+ str(k)
        ekU = cs.MX.sym(euName, 1) 
        w+=[ekU]
        lbw+=[0]
        ubw+=[np.inf]
        Guess+=[0]
            
        G+=[Fk-Xk]
        lbg+=[0]
        ubg+=[0]

        ubU = self.UB_mz1
        lbU = self.UL_mz1

        G+=[Uk-cs.mtimes(Ck,ubU)]
        lbg+=[-cs.inf]
        ubg+=[0]

        G+=[cs.mtimes(Ck,lbU)-Uk]
        lbg+=[-cs.inf]
        ubg+=[0]
        
        QLower=self.QLower_mz1
        QUpper=self.QUpper_mz1

        #Cost function
        # J+=cs.mtimes(cs.transpose(ekL),cs.mtimes(QLower,ekL))+(1000*Ck)+cs.mtimes(cs.transpose(ekU),cs.mtimes(QUpper,ekU))+(-9000*Uk)
        # J+=cs.mtimes(QLower,ekL) + (1000*Ck) + cs.mtimes(QUpper,ekU) + (-9000*Uk)
        # J+=cs.mtimes(QLower,ekL) + (5*Ck) + cs.mtimes(QUpper,ekU) + (-50*Uk)

        J+=cs.mtimes(QLower,ekL) + (self.R_c*Ck) + cs.mtimes(QUpper,ekU) + (self.R_u*Uk)

        lowerZone = self.LZ_scaled_mz1
        upperZone = self.UZ_scaled_mz1

        G+=[Xk-upperZone-ekU]
        lbg+=[-cs.inf]
        ubg+=[0]

        G+=[lowerZone-ekL-Xk]
        lbg+=[-cs.inf]
        ubg+=[0]
        pass

    #Solve the optimization problem, using IPOPT
    nlp=dict(f=J, g=cs.vertcat(*G), x=cs.vertcat(*w))
    # opts = {"ipopt.print_level":1}
    # solver=cs.nlpsol('solver','ipopt',nlp)

    opts = {"ipopt.print_level":0, 'print_time': False, 'ipopt.sb': "yes"}
    solver=cs.nlpsol('solver','ipopt',nlp,opts)

    r=solver(lbx=lbw, ubx=ubw,x0=Guess,lbg=lbg,ubg=ubg)
    opt_vals = r['x'].full().ravel()
    opt_vals = opt_vals.tolist()

    solver_status = solver.stats()['return_status']
    if solver_status == 'Solve_Succeeded' and stage=='Intermediate Step':
        dual_vars = r['lam_g'].full().ravel() # Based on the version of IPOPT used, the dual variables will have to be negated!
        print(np.max(np.abs(dual_vars)))
        # print("Dual variables: ", dual_vars)
        self.dual_vars_opt.append(dual_vars.tolist())
        self.opt_u.append(self.obtain_irrig_amount(opt_vals,5))
        self.opt_x.append(self.obtain_soil_moisture(opt_vals,5)[1])
        self.opt_x_sc.append(self.obtain_soil_moisture(opt_vals,5)[0])
        self.opt_eps_lbd.append(self.obtain_slack_vars(opt_vals,5)[0])
        self.opt_eps_ubd.append(self.obtain_slack_vars(opt_vals,5)[1])

        # we update the bounds for the feasible case of the subproblem    
        self.ubd_cur = r['f'].full().ravel()[0]
        self.ubd = min(self.ubd_cur,self.ubd_prev)
        self.ubd_prev = self.ubd
        self.x = self.obtain_soil_moisture(opt_vals,5)[1]
        self.u = self.obtain_irrig_amount(opt_vals,5)
        self.u_guess = self.obtain_irrig_amount(opt_vals,5) # we update the guess for the next iteration!
        self.x_guess = self.obtain_soil_moisture(opt_vals,5)[0] # we update the guess for the next iteration!
        self.cut_order.append("optimality_cut")
    
    elif solver_status =='Solve_Succeeded' and stage != 'Intermediate Step':
        # use the optimal values of the complicating variables to find the values of the non-complicating variables
        self.x = self.obtain_soil_moisture(opt_vals,5)[1]
        self.u = self.obtain_irrig_amount(opt_vals,5)
    else:
        w = []
        J = 0
        lbw = []
        ubw = []
        Guess = []
        G = []
        lbg = []
        ubg = []
        pastStateValues = []
        pastInputValues = []
    
        for j in range(self.sequenceLength):
            xname='X' + str(j)
            Xk=cs.MX.sym(xname, 1)
            w+=[Xk]
            lbw+=[self.currentStates[j]]
            ubw+=[self.currentStates[j]]
            Guess+=[self.currentStates[j]]
            pastStateValues.append(Xk)
            pass
        
        for k in range(self.sequenceLength - 1):
            uname = 'U' +str(k)
            Uk = cs.MX.sym(uname, 1)
            w+= [Uk]
            lbw+= [self.previousInputs[k]]
            ubw+= [self.previousInputs[k]]
            Guess+= [self.previousInputs[k]]
        
            cname = 'C' + str(k)
            ck = cs.MX.sym(cname, 1)
            w+= [ck]
            lbw+=[1]
            ubw+=[1]
            Guess+=[1]
            pastInputValues.append(Uk)
            pass 

        for k in range(1, self.prediction_horizon+1):
            Uname = 'u' + str(k-1)
            Uk = cs.MX.sym(Uname, 1)    
            w+= [Uk]
            lbw+= [-cs.inf]
            ubw+= [cs.inf]
            Guess+= [self.u_guess[k-1]]
            pastInputValues.append(Uk)

            cname = 'C' + str(k-1)
            Ck = cs.MX.sym(cname,1)
            w+=[Ck]
            lbw+=[self.c[k-1]]
            ubw+=[self.c[k-1]]
            Guess+=[self.c[k-1]]
        
            currentStateValues= pastStateValues[k-1 : (k-1) + self.sequenceLength]
            currentInputValues= pastInputValues[k-1 : (k-1) + self.sequenceLength]   
            currentCropCoefficient= self.cropCoeff[k-1 : (k-1) + self.sequenceLength]
            currentRefEvap = self.refEvap[k-1 : (k-1) + self.sequenceLength]
            currentRootingDepths = self.rooting_depths[k-1: (k-1) + self.sequenceLength]
            currentLAI_factors = self.lai_factors[k-1: (k-1) + self.sequenceLength]
            currentRainValues = self.rain[k-1 : (k-1) + self.sequenceLength] #Irrespective of the irrigation decision, the rain should have an effect on the system!

            currentPrecipValues_unscaled = []
            assert len(currentInputValues) == len(currentRainValues)
            for i in range(len(currentRainValues)):
                currentPrecipValues_unscaled.append(currentInputValues[i]+currentRainValues[i])
                pass 
        
            currentPrecipValues_scaled = []
            for i in range(len(currentRainValues)):
                currentPrecipValues_scaled.append(self.scale_irrig_amount((currentPrecipValues_unscaled[i])))
                pass 

            Fk = optimizeInputModel_automated_mz1(currentStateValues,currentPrecipValues_scaled,currentCropCoefficient,currentRefEvap,currentRootingDepths,currentLAI_factors)

            Xname = 'X' +str(k)
            Xk=cs.MX.sym(Xname, 1)
            w+=[Xk]
            lbw+=[-cs.inf]
            ubw+=[cs.inf]
            Guess+=[self.x_guess[k-1]]
            pastStateValues.append(Xk)
        
            elName = 'el' + str(k)
            ekL = cs.MX.sym(elName, 1) 
            w+=[ekL]
            lbw+=[0]
            ubw+=[cs.inf]
            Guess+=[0]
        
            euName = 'el'+ str(k)
            ekU = cs.MX.sym(euName, 1)
            w+=[ekU]
            lbw+=[0]
            ubw+=[np.inf]
            Guess+=[0]
            
            G+=[Fk - Xk]
            lbg+=[0]
            ubg+=[0]
        
            if self.rooting_depths[4+k-1] == 0:
                ubU = self.UB_mz1
                lbU = self.UL_mz1
            else:
                ubU = self.UB_mz1_mod
                lbU = self.UL_mz1_mod

            alLName_u = 'alL_u' + str(k)
            al_lbd_u = cs.MX.sym(alLName_u, 1)
            w+=[al_lbd_u]
            lbw+=[0]
            ubw+=[cs.inf]
            Guess+=[0]
                    
            alUName_u = 'alU_u' + str(k)
            al_ubd_u = cs.MX.sym(alUName_u, 1) 
            w+=[al_ubd_u]
            lbw+=[0]
            ubw+=[cs.inf]
            Guess+=[0]
        
            G+=[Uk-cs.mtimes(Ck, ubU)-al_ubd_u]
            lbg+=[-cs.inf]
            ubg+=[0]
  
            G+=[cs.mtimes(Ck, lbU)-Uk-al_lbd_u]
            lbg+=[-cs.inf]
            ubg+=[0]

            lowerZone = self.LZ_scaled_mz1 
            upperZone = self.UZ_scaled_mz1

            alLName_x = 'alL_x' + str(k)
            al_lbd_x = cs.MX.sym(alLName_x, 1) 
            w+=[al_lbd_x] 
            lbw+=[0]
            ubw+=[cs.inf]
            Guess+=[0]
                    
            alUName_x = 'alU_x' + str(k)
            al_ubd_x = cs.MX.sym(alUName_x, 1) 
            w+=[al_ubd_x]
            lbw+=[0]
            ubw+=[cs.inf]
            Guess+=[0]

            G+=[Xk-upperZone-ekU-al_ubd_x]
            lbg+=[-cs.inf]
            ubg+=[0]

            G+=[lowerZone-ekL-Xk-al_lbd_x]
            lbg+=[-cs.inf]
            ubg+=[0]

            J+=al_lbd_u+al_ubd_u+al_lbd_x+al_ubd_x  
            nlp=dict(f=J, g=cs.vertcat(*G), x=cs.vertcat(*w))
            # opts = {"ipopt.print_level":0}
            opts = {"ipopt.print_level":0, 'print_time': False, 'ipopt.sb': "yes"}
            solver=cs.nlpsol('solver','ipopt',nlp,opts)
            # solver=cs.nlpsol('solver','ipopt',nlp)
            r=solver(lbx=lbw, ubx=ubw,x0=Guess,lbg=lbg,ubg=ubg)
            opt_vals = r['x'].full().ravel()
            dual_vars = r['lam_g'].full().ravel()
            self.dual_vars_feas.append(dual_vars.tolist())
            self.feas_u.append(self.obtain_irrig_amount(opt_vals,9))
            self.feas_x.append(self.obtain_soil_moisture(opt_vals,9)[1])
            self.feas_x_sc.append(self.obtain_soil_moisture(opt_vals,9)[0])
            self.feas_eps_lbd.append(self.obtain_slack_vars(opt_vals,9)[0])
            self.feas_eps_ubd.append(self.obtain_slack_vars(opt_vals,9)[1]) 
            self.cut_order.append("feasibility_cut")                                                                                         
            pass
        pass