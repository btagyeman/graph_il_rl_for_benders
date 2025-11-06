import numpy as np
import tensorflow as tf
from core_agent.networks import ActorNetwork, CriticNetwork
import tensorflow_probability as tfp
from tensorflow import keras

# # GAE computation
def compute_gae(rewards, values, next_values, dones, gamma=0.99, lam=0.95):
    rewards = np.array(rewards, dtype=np.float32)
    values = np.array(values, dtype=np.float32)
    next_values = np.array(next_values, dtype=np.float32)
    dones = np.array(dones, dtype=np.float32)

    deltas = rewards + gamma * next_values * (1 - dones) - values
    advantages = []
    advantage = 0
    for delta, done in zip(reversed(deltas), reversed(dones)):
        if done:
            advantage = 0
        advantage = delta + gamma * lam * advantage
        advantages.insert(0, advantage)
    return tf.convert_to_tensor(advantages, dtype=tf.float32)

def compute_probs_batches(states, actor):
    x_all, a_all, e_all, i_all = states  # Each is a (N, ...) array
    probs_batch =[]
    for n in range(len(x_all)):
        x = x_all[n]
        a = a_all[n]
        e = e_all[n]
        i = i_all[n]
        state = [x, a, e, i]
        probs = actor(state)
        probs_batch.append(probs.numpy().squeeze())
    return np.array(probs_batch)

def compute_logits_batches_tf(states, actor):
    x_all, a_all, e_all, i_all = states
    logits_list = []
    for idx in range(len(x_all)):
        state = [x_all[idx], a_all[idx], e_all[idx], i_all[idx]]
        logits = actor(state)  # Tensor
        # probs = tf.nn.sigmoid(logits)
        # probs = tf.clip_by_value(probs, 1e-6, 1 - 1e-6)  # Avoids log(0)
        # probs = tf.clip_by_value(probs, 1e-8, 1 - 1e-8)  # Avoids log(0)
        logits_list.append(logits)
    return tf.concat(logits_list, axis=0)

def compute_vals_batches(states, critic):
    x_all, a_all, e_all, i_all = states  # Each is a (N, ...) array
    vals_batch =[]
    for n in range(len(x_all)):
        x = x_all[n]
        a = a_all[n]
        e = e_all[n]
        i = i_all[n]
        state = [x, a, e, i]
        val = critic(state)
        vals_batch.append(val.numpy().squeeze())
    return np.array(vals_batch)


def compute_probs_batches_tf(states, actor):
    x_all, a_all, e_all, i_all = states
    probs_list = []
    for idx in range(len(x_all)):
        state = [x_all[idx], a_all[idx], e_all[idx], i_all[idx]]
        probs = actor(state)  # Tensor
        probs = tf.clip_by_value(probs, 1e-6, 1 - 1e-6)  # Avoids log(0)
        # probs = tf.clip_by_value(probs, 1e-8, 1 - 1e-8)  # Avoids log(0)
        probs_list.append(probs)
    return tf.concat(probs_list, axis=0)

def compute_vals_batches_tf(states, critic):
    x_all, a_all, e_all, i_all = states
    vals_list = []
    for idx in range(len(x_all)):
        state = [x_all[idx], a_all[idx], e_all[idx], i_all[idx]]
        val = critic(state)  # Tensor
        vals_list.append(val)
    return tf.concat(vals_list, axis=0)

class PPOAgent:
    def __init__(self,gamma=0.99, lam=0.97, clip_ratio=0.25):
    # def __init__(self,gamma=0.99, lam=0.97, clip_ratio=0.10):
        self.gamma = gamma
        self.lam = lam
        self.clip_ratio = clip_ratio
        self.actor = ActorNetwork()
        self.critic = CriticNetwork()
        # self.actor.load_weights("imitation_learning_agent/mh_il_agent_model/variables/variables") # unnormalized
        # self.actor.load_weights("imitation_learning_agent/mh_il_agent_model_norm/variables/variables") # normalized
        # self.optimizer = keras.optimizers.Adam(learning_rate=1e-20)
        self.optimizer = keras.optimizers.Adam(learning_rate=5e-5)

    # def choose_action(self, state):
    #     probs = self.actor(state)  # shape (1, 3)
    #     probs = tf.clip_by_value(probs, 1e-6, 1 - 1e-6)
    #     dist = tfp.distributions.Bernoulli(probs=probs)
    #     action = dist.sample()
    #     log_prob = dist.log_prob(action)
    #     value = self.critic(state)
    #     return action.numpy().squeeze(), log_prob.numpy().squeeze(), value.numpy().squeeze().item(), probs.numpy().squeeze()
    
    def choose_action(self, state):
        logits = self.actor(state) 
        dist = tfp.distributions.Bernoulli(logits=logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        value = self.critic(state)
        probs = tf.sigmoid(logits)  # Optional: for thresholding or analysis
        return action.numpy().squeeze(), log_prob.numpy().squeeze(), value.numpy().squeeze().item(), probs.numpy().squeeze()
    #     return (
    #     action.numpy().squeeze(),
    #     log_prob.numpy().squeeze(),
    #     value.numpy().squeeze().item(),
    #     probs.numpy().squeeze(),
    # )

    # def choose_action(self, state, low_thresh=0.05, high_thresh=0.95):
    #     """
    #     - probs >= high_thresh → 1 (confident assignment)
    #     - probs <= low_thresh  → 0 (confident rejection)
    #     - else → random sample (encourages exploration in uncertain regions)
    #     """
    #     probs = self.actor(state)  # shape (1, 3)
    #     probs = tf.clip_by_value(probs, 1e-6, 1 - 1e-6)  # Avoids log(0)
    #     # probs = tf.clip_by_value(probs, 1e-8, 1 - 1e-8)  # Avoids log(0)
    #     # print(probs)
    #     value = self.critic(state)
    #     probs_np = probs.numpy().squeeze()
    #     actions = []
    #     log_probs = []

    #     for p in probs_np:
    #         if p >= high_thresh:
    #             action = 1
    #             log_prob = np.log(p + 1e-8)
    #         elif p <= low_thresh:
    #             action = 0
    #             log_prob = np.log(1 - p + 1e-8)
    #         else:
    #             action = np.random.binomial(1, p)
    #             action = -1
    #             # log_prob = np.log(p + 1e-8) if action == 1 else np.log(1 - p + 1e-8)
    #             log_prob = -1
    #         actions.append(action)
    #         log_probs.append(log_prob)

    #     return (np.array(actions),np.array(log_probs),value.numpy().squeeze().item(),probs_np)
    
    def learn(self, memory, advantages, returns, batch_size=32, epochs=5):
        states, actions, log_probs_old, _, _, _, _ = memory.get_trajectories()

        # Unpack graph components
        x_all, a_all, e_all, i_all = states  # Each is a (N, ...) array
        actions = np.array(actions)
        log_probs_old = np.array(log_probs_old)
        advantages = np.array(advantages)
        returns = np.array(returns)

        for _ in range(epochs):
            total_samples = len(actions)
            for start in range(0, total_samples, batch_size):
                end = min(start + batch_size, total_samples)

                # Slice the graph components for this batch
                batch_x = x_all[start:end]
                batch_a = a_all[start:end]
                batch_e = e_all[start:end]
                batch_i = i_all[start:end]

                batch_states = [batch_x, batch_a, batch_e, batch_i]

                batch_actions = actions[start:end]
                batch_log_probs_old = log_probs_old[start:end]
                batch_advantages = advantages[start:end]
                batch_returns = returns[start:end]

                with tf.GradientTape(persistent=True) as tape:
                    # probs = self.actor(batch_states)
                    # probs = compute_probs_batches_tf(batch_states, self.actor)
                    logits = compute_logits_batches_tf(batch_states, self.actor)
                    # print(probs)
                    # dist = tfp.distributions.Bernoulli(probs=probs)
                    dist = tfp.distributions.Bernoulli(logits=logits)
                    log_probs = dist.log_prob(batch_actions)
                    entropy = dist.entropy()

                    ratio = tf.exp(tf.reduce_sum(log_probs - batch_log_probs_old, axis=1))
                    # print(ratio)
                    clipped_ratio = tf.clip_by_value(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio)
                    actor_loss = -tf.reduce_mean(tf.minimum(ratio * batch_advantages,
                                                        clipped_ratio * batch_advantages))

                    # values_pred = self.critic(batch_states)
                    values_pred = compute_vals_batches_tf(batch_states, self.critic)
                    critic_loss = tf.reduce_mean(tf.square(batch_returns - tf.squeeze(values_pred)))
                    critic_loss = critic_loss

                    total_loss = actor_loss + 0.5 * critic_loss - 0.01 * tf.reduce_mean(entropy)

                actor_grads = tape.gradient(actor_loss, self.actor.trainable_variables)
                # print(actor_grads)
                critic_grads = tape.gradient(critic_loss, self.critic.trainable_variables)
                self.optimizer.apply_gradients(zip(actor_grads, self.actor.trainable_variables))
                self.optimizer.apply_gradients(zip(critic_grads, self.critic.trainable_variables))
                del tape
        memory.clear()