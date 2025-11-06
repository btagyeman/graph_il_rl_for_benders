import casadi as cs
from pyomo.environ import *
import numpy as np 
import sys
sys.path.append('../../trained_lstm_models')
from lstm_mz1 import optimizeInputModel_automated_mz1
def optimize_lagrangian(self, dual_variables, guesses):
    # This is basically an unconstrained optimization problem
    w = []
    J = 0
    lbw = []
    ubw = []
    Guess = []
    G = [] # Basically we do not have any constraints!
    lbg = []
    ubg = []
    pastStateValues = []
    pastInputValues = []
    guesses_x = guesses['x']
    guesses_u = guesses['u']
    guesses_e_lbd = guesses['e_lbd']
    guesses_e_ubd = guesses['e_ubd']
    idx_state = np.arange(0,(self.prediction_horizon*5),5).tolist()
    idx_u_ubd = np.arange(1,(self.prediction_horizon*5),5).tolist()
    idx_u_lbd = np.arange(2,(self.prediction_horizon*5),5).tolist()
    idx_x_ubd = np.arange(3,(self.prediction_horizon*5),5).tolist()
    idx_x_lbd = np.arange(4,(self.prediction_horizon*5),5).tolist()  
    
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
        pastInputValues.append(Uk)
        pass 

    for k in range(1, self.prediction_horizon+1):
        Uname = 'u' + str(k-1)
        Uk = cs.MX.sym(Uname, 1)    
        w+= [Uk]
        lbw+=[self.UL_mz1]
        ubw+=[self.UB_mz1]
        # Guess+= [guesses_u[k-1]]
        Guess+=[self.guessesU[k-1]]
        pastInputValues.append(Uk)
        
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

        Fk = optimizeInputModel_automated_mz1(currentStateValues,currentPrecipValues_scaled, 
                currentCropCoefficient,currentRefEvap,currentRootingDepths,currentLAI_factors)

        Xname = 'X' +str(k)
        Xk=cs.MX.sym(Xname, 1)
        w+=[Xk]
        # lbw+=[-cs.inf]
        # ubw+=[cs.inf]
        lbw+=[0]
        ubw+=[1]
        # Guess+=[guesses_x[k-1]]
        Guess+=[self.guessesX[k-1]]
        pastStateValues.append(Xk)
        
        elName = 'el' + str(k)
        ekL = cs.MX.sym(elName, 1)  
        w+=[ekL]
        lbw+=[0]
        ubw+=[cs.inf]
        # Guess+=[guesses_e_lbd[k-1]]
        Guess+=[0]
        
        euName = 'el'+ str(k)
        ekU = cs.MX.sym(euName, 1) 
        w+=[ekU]
        lbw+=[0]
        ubw+=[np.inf]
        # Guess+=[guesses_e_ubd[k-1]]
        Guess+=[0]

        J += cs.mtimes(dual_variables[idx_state[k-1]], Fk-Xk)   
        J += cs.mtimes(dual_variables[idx_u_ubd[k-1]], Uk)
        J += cs.mtimes(dual_variables[idx_u_lbd[k-1]],-Uk)
        
        QLower=self.QLower_mz1
        QUpper=self.QUpper_mz1

        #Cost function
        J+=cs.mtimes(cs.transpose(ekL),cs.mtimes(QLower,ekL))  + \
                    cs.mtimes(cs.transpose(ekU),cs.mtimes(QUpper,ekU)) + (-9000*Uk)

        # J+=cs.mtimes(cs.transpose(ekL),cs.mtimes(QLower,ekL))  + \
        #    cs.mtimes(cs.transpose(ekU),cs.mtimes(QUpper,ekU)) \
        #     + 9000*cs.mtimes(cs.transpose(Uk),Uk)  

        J+=cs.mtimes(dual_variables[idx_x_ubd[k-1]],Xk-self.UZ_scaled_mz1-ekU)
        J+=cs.mtimes(dual_variables[idx_x_lbd[k-1]],self.LZ_scaled_mz1-ekL-Xk)                                                                                                                          
        pass

    #Solve the optimization problem, using IPOPT
    nlp=dict(f=J, g=cs.vertcat(*G), x=cs.vertcat(*w))
    opts = {"ipopt.print_level":0}
    solver=cs.nlpsol('solver','ipopt',nlp,opts)
    r=solver(lbx=lbw, ubx=ubw,x0=Guess,lbg=lbg,ubg=ubg)
    solver_status = solver.stats()['return_status']
    print("Solver status for the optimization of the Lagrangian: ", solver_status)
    cost = r['f'].full().ravel()[0]
    return cost
