import tensorflow as tf 
from spektral.layers import GlobalSumPool, ECCConv
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense
import numpy as np
x_mean = np.load('./il_agent/x_mean.npy')
x_std = np.load('./il_agent/x_std.npy')
epsilon = 1e-8
e_mean = np.load('./il_agent/e_mean.npy')
e_std = np.load('./il_agent/e_std.npy')
# The actor network
class ActorNetwork(tf.keras.Model):
    def __init__(self, num_outputs=14):
        super().__init__()
        self.conv1 = ECCConv(64, activation='relu')
        self.conv2 = ECCConv(64, activation='relu')
        self.pool = GlobalSumPool()
        self.dense1 = Dense(64, activation='relu')
        self.output_heads = [Dense(1, activation='sigmoid') for _ in range(num_outputs)]

    def call(self, inputs):
        x, a, e, i = inputs
        x = (x - x_mean) / (x_std + epsilon)
        e = (e - e_mean) / (e_std + epsilon)
        x = self.conv1([x, a, e])
        x = self.conv2([x, a, e])
        x = self.pool([x, i])
        x = self.dense1(x)
        outputs = [head(x) for head in self.output_heads]  
        return tf.concat(outputs, axis=-1)

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
        x = (x - x_mean) / (x_std + epsilon)
        e = (e - e_mean) / (e_std + epsilon)
        x = self.conv1([x, a, e])
        x = self.conv2([x, a, e])
        x = self.pool([x, i])
        x = self.dense1(x)
        y = self.out_y(x)
        return y
