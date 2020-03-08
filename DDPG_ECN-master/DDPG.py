import matplotlib.pyplot as plt
import tensorflow as tf
import argparse
import keras.backend as K
from keras.layers import Dense, BatchNormalization
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras.callbacks import History
from keras import optimizers
import random
import numpy as np
from collections import deque

from Actor import Actor
from Critic import Critic
history = History()

# Deep Deterministc Policy Gradient Agent
class DDPGAgent:
    def __init__(self, state_size=28, action_size=2, gamma=0.9, learning_rate_actor=0.0001, learning_rate_critic=0.01, tau=0.001, action_max=[1000, 2], batch_size=32):
        self.state_size = state_size
        self.action_size = action_size
        self.action_max = action_max
        self.batch_size = batch_size
        self.memory = deque(maxlen=5000)
        self.gamma = gamma   # discount rate
        self.learning_rate_actor = learning_rate_actor      # learning rate
        self.learning_rate_critic = learning_rate_critic
        self.tau = tau      # target transfer factor
        self.gpu_options = tf.GPUOptions()
        self.config = tf.ConfigProto(gpu_options=self.gpu_options)
        self.config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=self.config)
        K.set_session(self.sess)
        self.actor = Actor(state_size=self.state_size, action_size=self.action_size, learning_rate=self.learning_rate_actor, tau=self.tau, sess=self.sess, batch_size=self.batch_size, action_max=self.action_max)
        self.critic = Critic(state_size=self.state_size, action_size=self.action_size, learning_rate=self.learning_rate_critic, gamma=self.gamma, tau=self.tau, sess=self.sess, batch_size=self.batch_size)
        self.grad_avg = 0
        self.grad_a = []
        self.critic_loss_a = []
        #self.critic_2 = Critic_2(self.state_size, self.action_size, self.learning_rate_critic, self.gamma, self.tau, self.sess)

    def policy_action(self, state):
        '''
        Actor predicts new action
        :param state:
        :return: action
        '''
        return self.actor.predict(state)[0]

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        states = np.asarray([e[0] for e in minibatch])
        actions = np.asarray([e[1] for e in minibatch])
        rewards = np.asarray([e[2] for e in minibatch])
        next_states = np.asarray([e[3] for e in minibatch])

        states = np.asarray(states).reshape(batch_size, self.state_size)
        actions = np.asarray(actions).reshape(batch_size, self.action_size)
        rewards = np.asarray(rewards).reshape(batch_size, 1)
        next_states = np.asarray(next_states).reshape(batch_size, self.state_size)
        tar_pre = self.actor.target_predict(next_states)
        Qvals = self.critic.target_predict(next_states, tar_pre)
        Q_primes = rewards + (self.gamma * Qvals)           # Bellman equation
        self.update_models(states, actions, Q_primes)

    def update_models(self, states, actions, critic_target):
        '''
        Update actor and critic networks from sampled experience
        :param states:
        :param actions:
        :param critic_target:
        :return:
        '''
        loss = self.critic.train_on_batch(states, actions, critic_target)      # Train Critic
        self.critic_loss_a.append(loss)
        # loss = np.sum(-np.log10(loss), axis=0)
        act = self.actor.predict(states)                # Q Value Gradient under Current Policy
        grads = self.critic.gradients(states, act)          # actor loss

        self.grad_avg += np.sum(np.log10(np.absolute(grads)), axis=0)/self.batch_size
        self.grad_a = np.append(self.grad_a, np.sum(np.absolute(grads), axis=0)/self.batch_size, axis=0)
        # print('grad_a:', self.grad_a)

        self.actor.train_2(states, grads.reshape((-1, self.action_size)))               # Train actor

        self.actor.transfer_to_actor_model()        # Transfer weights to target networks at rate tau
        self.critic.transfer_to_critic_model()

    def remember(self, state, action, reward, next_state):
        self.memory.append((state, action, reward, next_state))

    def save_weights(self, directory, params):
        path_actor = directory + 'Weights' + params + '_LR'.format(self.learning_rate_actor)
        path_critic = directory + 'Weights' + params +'_LR'.format(self.learning_rate_critic)
        self.actor.save(path_actor)
        self.critic.save(path_critic)

    def load_weights(self, path_actor, path_critic):
        self.actor.load_weigths(path_actor)
        self.critic.load_weights(path_critic)

    def load_model(self, path_actor, path_critic):
        self.actor.model.load_model(path_actor)
        self.critic.model.load_model(path_critic)





        #     target = reward + self.gamma * \
        #                np.amax(self.target_model.predict(next_state)[0])
        #     target_f = self.model.predict(state)
        #     target_f[0][action] = target
        #     self.model.fit(state, target_f, epochs=1, verbose=0, callbacks=[history])
        #
        # for target_param, param in zip(self.actor_model.model.get_weights() )
        #
        # self.t_update_count += 1
        # self.loss = np.append(self.loss, history.history['loss'][0])
        # self.loss_avg += history.history['loss'][0]
        #
        # if self.t_update_count >= self.t_update_thresh:
        #     self.update_target_model()
        # if self.epsilon > self.epsilon_min:
        #     self.epsilon *= self.epsilon_decay