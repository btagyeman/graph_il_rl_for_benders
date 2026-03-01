import casadi as cs
from pyomo.environ import *
import numpy as np 
import sys
sys.path.append('../../trained_lstm_models')
from lstm_mz1 import optimizeInputModel_automated_mz1

def original_milp(self):
    w = []
    J = 0
    lbw = []
    ubw = []
    discrete = []
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
        discrete+=[False]
        lbw+=[self.currentStates[j]]
        ubw+=[self.currentStates[j]]
        Guess+= [self.currentStates[j]]
        pastStateValues.append(Xk)
        pass
        
    for k in range(self.sequenceLength - 1):
        uname = 'U' +str(k)
        Uk = cs.MX.sym(uname, 1)
        w+= [Uk]
        discrete+=[False]
        lbw+=[self.previousInputs[k]]
        ubw+=[self.previousInputs[k]]
        Guess+= [self.previousInputs[k]]
        
        cname = 'C' + str(k)
        ck = cs.MX.sym(cname, 1)
        w+= [ck]
        lbw+=[1]
        ubw+=[1]
        discrete+=[True]
        Guess+=[1]
        pastInputValues.append(Uk)
        pass 

    for k in range(1, self.prediction_horizon+1):
        Uname = 'u' + str(k-1)
        Uk = cs.MX.sym(Uname, 1)    
        w+= [Uk]
        discrete+=[False]
        lbw+=[-cs.inf]
        ubw+=[cs.inf]
        Guess+= [self.u_guess[k-1]]
        pastInputValues.append(Uk)

        cname = 'C' + str(k-1)
        Ck = cs.MX.sym(cname,1)
        w+=[Ck]
        discrete+=[True]
        lbw+=[0]
        ubw+=[1]
        Guess+=[self.c_guess[k-1]]
        
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

        Fk = optimizeInputModel_automated_mz1(currentStateValues, currentPrecipValues_scaled, 
                    currentCropCoefficient, currentRefEvap, currentRootingDepths, currentLAI_factors)

        Xname = 'X' +str(k)
        Xk=cs.MX.sym(Xname, 1)
        w+=[Xk]
        discrete+=[False]
        lbw+=[-cs.inf]
        ubw+=[cs.inf]
        Guess+=[self.x_guess[k-1]]
        pastStateValues.append(Xk)
        
        elName = 'el' + str(k)
        ekL = cs.MX.sym(elName, 1)  
        w+=[ekL]
        discrete+=[False]
        lbw+=[0]
        ubw+=[cs.inf]
        Guess+=[0]
        
        euName = 'el'+ str(k)
        ekU = cs.MX.sym(euName, 1) 
        w+=[ekU]
        discrete+=[False]
        lbw+=[0]
        ubw+=[np.inf]
        Guess+=[0]
            
        G+=[Fk - Xk]
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
        J+=cs.mtimes(cs.transpose(ekL),cs.mtimes(QLower,ekL))  + (1000*Ck) + \
                    cs.mtimes(cs.transpose(ekU),cs.mtimes(QUpper,ekU)) + (-9000*Uk)                                                                                                                            
    
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
    opts = {"discrete":discrete}
    solver=cs.nlpsol('solver','bonmin',nlp,opts)

    r=solver(lbx=lbw, ubx=ubw,x0=Guess,lbg=lbg,ubg=ubg)
    opt_vals = r['x'].full().ravel()
    opt_vals = opt_vals.tolist()

    opt_u = self.obtain_irrig_amount(opt_vals,5)
    opt_x = self.obtain_soil_moisture(opt_vals,5)[1]
    opt_cost = r['f'].full().ravel()[0]

    print("The optimal cost is: ", opt_cost)
    print("The optimal irrigation is: ", opt_u)
    print("The optimal soil moisture is: ", opt_x)
    return