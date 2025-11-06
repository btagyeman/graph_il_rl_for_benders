import numpy as np
# class PPOMemory:
#     def __init__(self):
#         self.states = []
#         self.actions = []
#         self.log_probs = []
#         self.rewards = []
#         self.dones = []
#         self.values = []
#         self.next_values = []

#     def store_transitions(self, state, action, log_prob, reward, done, value):
#         self.states.append(state)  # state = (x, a, e, i)
#         self.actions.append(action)
#         self.log_probs.append(log_prob)
#         self.rewards.append(reward)
#         self.dones.append(done)
#         self.values.append(value)

#     def store_next_values(self, next_value):
#         # self.next_values.append(next_value)
#         self.next_values.extend(next_value)

#     def clear(self):
#         self.states.clear()
#         self.actions.clear()
#         self.log_probs.clear()
#         self.rewards.clear()
#         self.dones.clear()
#         self.values.clear()
#         self.next_values.clear()

#     def get_trajectories(self):
#         return (
#             self.states,
#             self.actions,
#             self.log_probs,
#             self.rewards,
#             self.dones,
#             self.values,
#             self.next_values,
#         )

class PPOMemory:
    def __init__(self):
        self.x = []
        self.a = []
        self.e = []
        self.i = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.dones = []
        self.values = []
        self.next_values = []

    def store_transitions(self, state, action, log_prob, reward, done, value):
        x, a, e, i = state
        self.x.append(x)
        self.a.append(a)
        self.e.append(e)
        self.i.append(i)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        # self.rewards.append(reward)
        self.dones.append(done)
        self.values.append(value)

    def store_rewards(self, reward):
        self.rewards.extend(reward)

    def store_next_values(self, next_value):
        # self.next_values.append(next_value)
        self.next_values.extend(next_value)

    def get_trajectories(self):
        states = (np.array(self.x), np.array(self.a), np.array(self.e), np.array(self.i))
        return states, self.actions, self.log_probs, self.rewards, self.dones, self.values, self.next_values

    def clear(self):
        self.__init__()
