import tensorflow as tf 
from spektral.layers import GlobalSumPool, ECCConv
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense
import numpy as np 

# load the scaling parameters for standardization
x_mean = np.load("il_agent/x_mean.npy")
x_std = np.load("il_agent/x_std.npy")

e_mean = np.load("il_agent/e_mean.npy")
e_std = np.load("il_agent/e_std.npy")

# # load the scaling parameters for standardization
# x_mean = np.load("x_mean.npy")
# x_std = np.load("x_std.npy")

# e_mean = np.load("e_mean.npy")
# e_std = np.load("e_std.npy")

epsilon = 1e-8

class ActorNetwork(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.conv1 = ECCConv(64, activation='relu')
        self.conv2 = ECCConv(64, activation='relu')
        self.pool = GlobalSumPool()
        self.dense1 = Dense(64, activation='relu')
        # self.out_y1 = Dense(1,activation='sigmoid')
        # self.out_y2 = Dense(1,activation='sigmoid')
        # self.out_y3 = Dense(1,activation='sigmoid')
        # self.out_y4 = Dense(1,activation='sigmoid')
        # self.out_y5 = Dense(1,activation='sigmoid')
        self.out_y1 = Dense(1)
        self.out_y2 = Dense(1)
        self.out_y3 = Dense(1)
        self.out_y4 = Dense(1)
        self.out_y5 = Dense(1)

    def call(self, inputs):
        x, a, e, i = inputs
        x = (x - x_mean) / (x_std + epsilon)  # standardize the node features
        e = (e - e_mean) / (e_std + epsilon)  # standardize the edge features
        x = self.conv1([x, a, e])
        x = self.conv2([x, a, e])
        x = self.pool([x, i])
        x = self.dense1(x)
        y1 = self.out_y1(x)
        y2 = self.out_y2(x)
        y3 = self.out_y3(x)
        y4 = self.out_y4(x)
        y5 = self.out_y5(x)
        return tf.concat([y1, y2, y3, y4, y5], axis=-1)  

# The critic network will be randomly initialized
class CriticNetwork(Model):
    def __init__(self):
        super().__init__()
        self.conv1 = ECCConv(64, activation='relu')
        self.conv2 = ECCConv(64, activation='relu')
        self.pool = GlobalSumPool()
        self.dense1 = Dense(64, activation='relu')
        self.out_y = Dense(1, activation='linear')

    def call(self, inputs):
        x, a, e, i = inputs
        x = (x - x_mean) / (x_std + epsilon) # standardize the node features
        e = (e - e_mean) / (e_std + epsilon) # standardize the edge features
        x = self.conv1([x, a, e])
        x = self.conv2([x, a, e])
        x = self.pool([x, i])
        x = self.dense1(x)
        y = self.out_y(x)
        return y