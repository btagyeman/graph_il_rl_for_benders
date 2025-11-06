#Imports
import numpy as np
import casadi as cs
#Load the weights for the LSTM model
#Layer 1
W_1 = np.loadtxt("../../trained_lstm_models/model_weights/K_1_mz1_vrd_lai.txt")
U_1 = np.loadtxt("../../trained_lstm_models/model_weights/RK_1_mz1_vrd_lai.txt")
b_1 = np.loadtxt("../../trained_lstm_models/model_weights/BR_1_mz1_vrd_lai.txt")
#Layer 2
W_2 = np.loadtxt("../../trained_lstm_models/model_weights/K_2_mz1_vrd_lai.txt")
U_2 = np.loadtxt("../../trained_lstm_models/model_weights/RK_2_mz1_vrd_lai.txt")
b_2 = np.loadtxt("../../trained_lstm_models/model_weights/BR_2_mz1_vrd_lai.txt")
#Dense layer
V_m = np.loadtxt("../../trained_lstm_models/model_weights/KD_mz1_vrd_lai.txt")
b_m = np.loadtxt("../../trained_lstm_models/model_weights/BD_mz1_vrd_lai.txt")
#Define the number of units in each LSTM layer, the sequence length and the number of outputs
unitsLayer_1 = 400
unitsLayer_2 = 400
sequenceLength = 5
numberOfFeatures = 6

#Obtain the weights for the various gates that make up the LSTM
#Layer 1
#The kernel matrices for the gates and the cell state
W_i_1 = W_1[:,:unitsLayer_1] # The input gate
W_f_1 = W_1[:,unitsLayer_1: unitsLayer_1*2] # The forget gate
W_c_1 = W_1[:, unitsLayer_1*2:unitsLayer_1*3] # The cell state
W_o_1 = W_1[:, unitsLayer_1*3:] # The output gate
#The recurrent kernel matrices for the gates and the cell state
U_i_1 = U_1[:,:unitsLayer_1] # The input gate
U_f_1 = U_1[:,unitsLayer_1:unitsLayer_1*2] # The forget gate
U_c_1 = U_1[:, unitsLayer_1*2:unitsLayer_1*3] # The cell state
U_o_1 = U_1[:, unitsLayer_1*3:] # The output gate
# The bias matrices for the gates and the cell state
b_i_1 = b_1[:unitsLayer_1] # The input gate
b_f_1 = b_1[unitsLayer_1:unitsLayer_1*2] # The forget gate
b_c_1 = b_1[unitsLayer_1*2:unitsLayer_1*3] #The cell state
b_o_1 = b_1[unitsLayer_1*3:] # The output gate
#Layer 2
#The kernel matrices for the gates and the cell state
W_i_2 = W_2[:,:unitsLayer_2] # The input gate
W_f_2 = W_2[:,unitsLayer_2: unitsLayer_2*2] # The forget gate
W_c_2 = W_2[:, unitsLayer_2*2:unitsLayer_2*3] # The cell state
W_o_2 = W_2[:, unitsLayer_2*3:] # The output gate
#The recurrent kernel matrices for the gates and the cell state
U_i_2 = U_2[:,:unitsLayer_2] # The input gate
U_f_2 = U_2[:,unitsLayer_2:unitsLayer_2*2] # The forget gate
U_c_2 = U_2[:, unitsLayer_2*2:unitsLayer_2*3] # The cell state
U_o_2 = U_2[:, unitsLayer_2*3:] # The output gate
# The bias matrices for the gates and the cell state
b_i_2 = b_2[:unitsLayer_2] # The input gate
b_f_2 = b_2[unitsLayer_2:unitsLayer_2*2] # The forget gate
b_c_2 = b_2[unitsLayer_2*2:unitsLayer_2*3] #The cell state
b_o_2 = b_2[unitsLayer_2*3:] # The output gate

def sigmoidFunction_casadi(x):
    return 1/(1+cs.exp(-x))

def sigmoidFunction_numpy(x):
    return 1/(1+np.exp(-x))

def sigmoidFunction_original(slope,x):
    return 1/(1+np.exp(-slope*x))

def optimizeInputModel_automated_mz1(pressureHeads, irrigationAmounts, cropCoefficient, referenceET, rooting_depth, lai_factor):  
    hiddenState_layer1_init = cs.MX.zeros(unitsLayer_1)
    hiddenState_layer2_init = cs.MX.zeros(unitsLayer_2)
    cellState_layer1_init = cs.MX.zeros(unitsLayer_1)
    cellState_layer2_init = cs.MX.zeros(unitsLayer_2)
    inputs = cs.MX.zeros((sequenceLength, numberOfFeatures))
    inputs[:,2] = cropCoefficient
    inputs[:,3] = referenceET
    inputs[:,4] = rooting_depth
    inputs[:,5] = lai_factor
    
    for j in range(sequenceLength):
        inputs[j,0] = pressureHeads[j]
        inputs[j,1] = irrigationAmounts[j]
        pass 

    for j in range(sequenceLength):
        f_1 = cs.mtimes(cs.transpose(W_f_1), cs.transpose(inputs[j,:]))  + cs.mtimes(cs.transpose(U_f_1), hiddenState_layer1_init) + b_f_1
        f_t_1 = sigmoidFunction_casadi(f_1)
        i_1 = cs.mtimes(cs.transpose(W_i_1), cs.transpose(inputs[j,:]))  + cs.mtimes(cs.transpose(U_i_1), hiddenState_layer1_init) + b_i_1
        i_t_1 = sigmoidFunction_casadi(i_1)
        o_1 = cs.mtimes(cs.transpose(W_o_1), cs.transpose(inputs[j,:]))  + cs.mtimes(cs.transpose(U_o_1), hiddenState_layer1_init) + b_o_1
        o_t_1 = sigmoidFunction_casadi(o_1)
        c_1 = cs.mtimes(cs.transpose(W_c_1), cs.transpose(inputs[j,:]))  + cs.mtimes(cs.transpose(U_c_1), hiddenState_layer1_init) + b_c_1
        c_t_1 = cs.tanh(c_1)
        C_t_1 = f_t_1 * cellState_layer1_init + i_t_1 * c_t_1
        h_t_1 = o_t_1 * cs.tanh(C_t_1)
        f_2 = cs.mtimes(cs.transpose(W_f_2), h_t_1)  + cs.mtimes(cs.transpose(U_f_2), hiddenState_layer2_init) + b_f_2
        f_t_2 = sigmoidFunction_casadi(f_2)
        i_2 = cs.mtimes(cs.transpose(W_i_2), h_t_1)  + cs.mtimes(cs.transpose(U_i_2), hiddenState_layer2_init) + b_i_2
        i_t_2 = sigmoidFunction_casadi(i_2)
        o_2 = cs.mtimes(cs.transpose(W_o_2), h_t_1)  + cs.mtimes(cs.transpose(U_o_2), hiddenState_layer2_init) + b_o_2
        o_t_2 = sigmoidFunction_casadi(o_2)
        c_2 = cs.mtimes(cs.transpose(W_c_2), h_t_1)  + cs.mtimes(cs.transpose(U_c_2), hiddenState_layer2_init) + b_c_2
        c_t_2 = cs.tanh(c_2)
        C_t_2 = f_t_2 * cellState_layer2_init  + i_t_2 *  c_t_2
        h_t_2 = o_t_2 *  cs.tanh(C_t_2)
        hiddenState_layer1_init = h_t_1
        cellState_layer1_init = C_t_1
        hiddenState_layer2_init = h_t_2
        cellState_layer2_init = C_t_2
    O_t = b_m + cs.mtimes(cs.transpose(V_m), h_t_2)
    return O_t

def getRNNmodel_numpy_mz1(pressureHead, irrigationAmount, cropCoefficient, referenceET, rooting_depth, lai_factor):
    hiddenState_layer1_init = np.zeros(unitsLayer_1)
    hiddenState_layer2_init = np.zeros(unitsLayer_2)
    cellState_layer1_init = np.zeros(unitsLayer_1)
    cellState_layer2_init = np.zeros(unitsLayer_2)
    inputs = np.zeros((sequenceLength, numberOfFeatures))
    inputs[:,0] = pressureHead
    inputs[:,1] = irrigationAmount
    inputs[:,2] = cropCoefficient
    inputs[:,3] = referenceET
    inputs[:,4] = rooting_depth
    inputs[:,5] = lai_factor
    
    for j in range(sequenceLength):
        f_1 = np.matmul(np.transpose(W_f_1), inputs[j])  + np.matmul(np.transpose(U_f_1), hiddenState_layer1_init) + np.transpose(b_f_1)
        f_t_1 = sigmoidFunction_numpy(f_1)
        i_1 = np.matmul(np.transpose(W_i_1), inputs[j])  + np.matmul(np.transpose(U_i_1), hiddenState_layer1_init) + np.transpose(b_i_1)
        i_t_1 = sigmoidFunction_numpy(i_1)
        o_1 = np.matmul(np.transpose(W_o_1), inputs[j])  + np.matmul(np.transpose(U_o_1), hiddenState_layer1_init) + np.transpose(b_o_1)
        o_t_1 = sigmoidFunction_numpy(o_1)
        c_1 = np.matmul(np.transpose(W_c_1), inputs[j])  + np.matmul(np.transpose(U_c_1), hiddenState_layer1_init) + np.transpose(b_c_1)
        c_t_1 = np.tanh(c_1)
        C_t_1 = np.multiply(f_t_1, cellState_layer1_init) + np.multiply(i_t_1, c_t_1)
        h_t_1 = np.multiply(o_t_1, np.tanh(C_t_1))
        f_2 = np.matmul(np.transpose(W_f_2), h_t_1)  + np.matmul(np.transpose(U_f_2), hiddenState_layer2_init) + np.transpose(b_f_2)
        f_t_2 = sigmoidFunction_numpy(f_2)
        i_2 = np.matmul(np.transpose(W_i_2), h_t_1)  + np.matmul(np.transpose(U_i_2), hiddenState_layer2_init) + np.transpose(b_i_2)
        i_t_2 = sigmoidFunction_numpy(i_2)
        o_2 = np.matmul(np.transpose(W_o_2), h_t_1)  + np.matmul(np.transpose(U_o_2), hiddenState_layer2_init) + np.transpose(b_o_2)
        o_t_2 = sigmoidFunction_numpy(o_2)
        c_2 = np.matmul(np.transpose(W_c_2), h_t_1)  + np.matmul(np.transpose(U_c_2), hiddenState_layer2_init) + np.transpose(b_c_2)
        c_t_2 = np.tanh(c_2)
        C_t_2 = np.multiply(f_t_2, cellState_layer2_init) + np.multiply(i_t_2, c_t_2)
        h_t_2 = np.multiply(o_t_2, np.tanh(C_t_2))
        hiddenState_layer1_init = h_t_1
        cellState_layer1_init = C_t_1
        hiddenState_layer2_init = h_t_2
        cellState_layer2_init = C_t_2
    O_t=np.transpose(b_m) + np.matmul(np.transpose(V_m), h_t_2)
    return O_t        
