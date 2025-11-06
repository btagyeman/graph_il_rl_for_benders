import numpy as np 
import casadi as cs
import scipy.sparse.linalg
from  common.estimator_design.temporal_spatial_ekf import spatialVariables_1D, temporalVariable
from common.estimator_design.richards_equation_ekf import RichardsPolar_1D, rz_moisture_alg,rz_moisture_sym,surface_moisture_alg,surface_moisture_sym

totalDepth, axialNodes, axialDistance, totalNodes = spatialVariables_1D()
samplingTime, samplingTimeInternal, internalTimeSteps, totTimeSteps = temporalVariable()


def evaluate_jacobian(head, irrigAmount, cropCoeff, refEvap, laiFactor, soilPars):
    head_sym = cs.SX.sym('h', totalNodes)
    irrigRate_sym = cs.SX.sym('u', 1)
    cropCoeff_sym = cs.SX.sym('c', 1)
    refEvap_sym = cs.SX.sym('e', 1)
    laiFactor_sym = cs.SX.sym('l', 1)
    symbolicModel = RichardsPolar_1D(head_sym, irrigRate_sym, cropCoeff_sym, refEvap_sym, laiFactor_sym, soilPars)
    jacHead = cs.jacobian(symbolicModel, head_sym)
    jacHeadFun = cs.Function('jacHeadFun', [head_sym, irrigRate_sym, cropCoeff_sym, refEvap_sym, laiFactor_sym],[jacHead])
    A_continuous = jacHeadFun(head, irrigAmount, cropCoeff, refEvap, laiFactor)
    return A_continuous



# def discreteTimeEKF(prevHead,predHead,irrigAmount,cropCoeff,refEvap,laiFactor,measSeq,prevCov,procNoiseCov,measNoiseCov,soilPars):
#     #Computing the jacobian of the output function with respect to the state(pressure head) and the C matrix
#     headSym = cs.SX.sym('x', totalNodes)
#     outputSym = rz_moisture_sym(headSym,soilPars)
#     outputJac = cs.jacobian(outputSym, headSym)
#     jacOutputFun = cs.Function('jacOutputFun', [headSym], [outputJac])
#     C_discrete = jacOutputFun(predHead)
#     #Obtaining the continuous and discrete time A matrices 
#     A_continuous = evaluate_jacobian(prevHead, irrigAmount, cropCoeff, refEvap, laiFactor, soilPars)
#     A_discrete = cs.DM(scipy.sparse.linalg.expm(np.array(samplingTimeInternal *A_continuous)))
    
#     # condNumber = np.linalg.cond(np.array(A_discrete))
    
#     #Updating the state covariance matrix based on the model of the system
#     P_prediction = cs.DM(np.mat(A_discrete)*np.mat(prevCov)*np.mat(A_discrete.T) + np.mat(procNoiseCov))

#     #Computing the Kalman gain
#     innovCov = cs.mtimes(C_discrete, cs.mtimes(P_prediction, np.transpose(C_discrete))) + measNoiseCov
#     KalmanGain = cs.mtimes(cs.mtimes(P_prediction, np.transpose(C_discrete)), np.linalg.inv(innovCov))

#     #Computing the estimate of the state in the EKF update step
#     innovations = measSeq -  rz_moisture_alg(predHead,soilPars)
#     head_update = predHead + cs.mtimes(KalmanGain,innovations)

#     #Computing the convariance matrix of the state in the EKF update step
#     P_update = cs.DM((np.mat(cs.DM.eye(totalNodes))-np.mat(KalmanGain)*np.mat(C_discrete))*np.mat(P_prediction))

#     #Converting results obtained from casadi to numpy
#     headEst = head_update.full().ravel()
#     Cov_headEst = P_update.full()
 
#     return [headEst, Cov_headEst]



def discreteTimeEKF(prevHead,predHead,irrigAmount,cropCoeff,refEvap,laiFactor,measSeq,prevCov,procNoiseCov,measNoiseCov,soilPars):
    #Computing the jacobian of the output function with respect to the state(pressure head) and the C matrix
    headSym = cs.SX.sym('x', totalNodes)
    # outputSym = rz_moisture_sym(headSym,soilPars)
    outputSym = surface_moisture_sym(headSym,soilPars)
    outputJac = cs.jacobian(outputSym, headSym)
    jacOutputFun = cs.Function('jacOutputFun', [headSym], [outputJac])
    C_discrete = jacOutputFun(predHead)
    #Obtaining the continuous and discrete time A matrices 
    A_continuous = evaluate_jacobian(prevHead, irrigAmount, cropCoeff, refEvap, laiFactor, soilPars)
    A_discrete = cs.DM(scipy.sparse.linalg.expm(np.array(samplingTimeInternal*A_continuous)))
    
    # condNumber = np.linalg.cond(np.array(A_discrete))
    
    #Updating the state covariance matrix based on the model of the system
    P_prediction = cs.DM(np.mat(A_discrete)*np.mat(prevCov)*np.mat(A_discrete.T) + np.mat(procNoiseCov))

    #Computing the Kalman gain
    innovCov = cs.mtimes(C_discrete, cs.mtimes(P_prediction, np.transpose(C_discrete))) + measNoiseCov
    KalmanGain = cs.mtimes(cs.mtimes(P_prediction, np.transpose(C_discrete)), np.linalg.inv(innovCov))

    #Computing the estimate of the state in the EKF update step
    # innovations = measSeq -  rz_moisture_alg(predHead,soilPars)
    innovations = measSeq -  surface_moisture_alg(predHead,soilPars)
    head_update = predHead + cs.mtimes(KalmanGain,innovations)

    #Computing the convariance matrix of the state in the EKF update step
    P_update = cs.DM((np.mat(cs.DM.eye(totalNodes))-np.mat(KalmanGain)*np.mat(C_discrete))*np.mat(P_prediction))

    #Converting results obtained from casadi to numpy
    headEst = head_update.full().ravel()
    Cov_headEst = P_update.full()
 
    return [headEst, Cov_headEst]