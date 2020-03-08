import keras.backend as K
import tensorflow as tf
from keras.initializers import RandomUniform
from keras.layers import Dense, Flatten, Input, Activation, concatenate, BatchNormalization, merge, Add
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras.callbacks import History
from keras import optimizers
import numpy as np
from collections import deque
history = History()

HIDDEN1_UNITS = 16
HIDDEN2_UNITS = 32

# Deep Deterministc Policy Gradient Agent
# Critic class
class Critic:
    def __init__(self, state_size=28, action_size=3, learning_rate=0.001, gamma=0.001, tau=0.001, sess=None, batch_size=32):
        self.state_size = state_size        # set state size
        self.action_size = action_size      # set action size
        self.memory = deque(maxlen=5000)
        self.gamma = gamma   # discount rate
        self.learning_rate = learning_rate  # learning rate
        self.tau = tau      # soft copy factor
        self.sess = sess       # tensorflow session
        self.batch_size = batch_size       # batch size
        # self.model = self._build_critic_model()
        # self.target_model = self._build_critic_model()
        self.model, self.action_input, self.state_input = self.create_network()     # create critic network
        self.target_model, self.target_action, self.target_state = self.create_network()        # create target network
        self.action_grads = tf.gradients(self.model.output, self.action_input)      # compute action gradients 
        print(self.action_grads)
        self.sess.run(tf.initialize_all_variables())           
        #self.action_grads = K.function([self.model.input[0], self.model.input[1]], K.gradients(self.model.output, [self.model.input[1]]))

    # def _build_critic_model(self):
    #     # Neural Net for Critic Model
    #     model = Sequential()
    #     model.add(Dense(16, input_dim=self.state_size + self.action_size, activation='sigmoid'))  # sigmoid
    #     model.add(Dense(32, activation='relu'))  # relu
    #     model.add(Dense(32, activation='relu'))
    #     #       model.add(Dense(32, activation='tanh'))
    #     model.add(Dense(self.state_size, activation='linear'))  # linear
    #     # sgd = optimizers.SGD(lr=self.learning_rate, decay=self.epsilon_decay, momentum=0.9, nesterov=True)
    #     # model.compile(loss='mse', optimizer=sgd)
    #     model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
    #     return model

    def create_network(self):
        '''
        create network
        '''
        S = Input(shape=[self.state_size])      # state input
        Bstate1 = BatchNormalization()(S)       # batch normalization
        A = Input(shape=[self.action_size])     # action input
        Baction1 = BatchNormalization()(A)      # batch normalization
        # state2 = Dense(16, activation='linear')(Bstate1)      #300     128       128   128 32
        # state3 = Dense(16, activation='relu')(state2)
        # state4 = Dense(16, activation='relu')(state3)
        state4 = Dense(4*self.state_size, activation='relu')(Bstate1)      # hidden layer for state stream
        state5 = Dense(4*self.state_size, activation='relu')(state4)        # 300     128       128   128 32
        # state2 = Dense(16, activation='linear')(S)
        # state3 = Dense(16, activation='relu')(state2)           #300   128    128  64 24        # state4 = Dense(512, activation='relu')(state3)
        # state4 = Dense(16, activation='relu')(state3)
        # state5 = Dense(16, activation='relu')(state4)
        # action2 = Dense(32, activation='linear')(Baction1)           #128   128   64 4
        # action3 = Dense(32, activation='relu')(action2)           #350     128   128   128 8
        # action4 = Dense(32, activation='relu')(action3)           #350     128   128   128 8
        action4 = Dense(5*self.state_size, activation='relu')(Baction1)           # hidden layers for action stream 
        action5 = Dense(5*self.state_size, activation='relu')(action4)            # 128   128   64 4
        # action2 = Dense(32, activation='linear')(A)
        # action3 = Dense(32, activation='relu')(action2)
        # action4 = Dense(32, activation='relu')(action3)
        # action5 = Dense(32, activation='relu')(action4)
        t1 = concatenate([action5, state5])         # concatenate state and action stream
        b3 = BatchNormalization()(t1)               # batch normalization
        # t2 = Dense(64, activation='relu')(b3)           # 350   512    128  254 32
        t2 = Dense(6*self.state_size, activation='relu')(b3)           # 350   512    128  254 32

        # t2 = Dense(64, activation='relu')(t1)
        # t3 = Dense(300, activation='relu', kernel_initializer=RandomUniform())(t2)
        # wl = Dense(self.action_size-1, activation='softmax')(t2)
        # blkl = Dense(1, activation='relu')(t2)
        # out = concatenate([blkl, wl])

        b4 = BatchNormalization()(t2)
        V = Dense(self.action_size, activation='linear', kernel_initializer=RandomUniform(seed=1))(b4)  # output layer
        model = Model(input=[S, A], output=V)       # create network model
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        return model, A, S

    def transfer_to_critic_model(self):     # copy critic network to target critic network
        W, target_W = self.model.get_weights(), self.target_model.get_weights()
        for i in range(len(W)):
            target_W[i] = self.tau * W[i] + (1 - self.tau) * target_W[i]
        self.target_model.set_weights(target_W)

    def target_predict(self, states, actions):  # compute Q-values
        return self.target_model.predict([np.asarray(states).reshape((self.batch_size, self.state_size)), np.asarray(actions)])

    def train_on_batch(self, states, actions, critic_target):  # train critic network
        #print(np.asarray([states, actions]).reshape((2, 1)))
        return self.model.train_on_batch([states, actions], critic_target)

    def gradients(self, states, actions):   # compute action gradient
        return self.sess.run(self.action_grads, feed_dict={
            self.state_input: states,
            self.action_input: actions
        })[0]
        #return self.action_grads([np.asarray(states), np.asarray(actions)])

    def save(self, path):
        self.model.save_weights(path + '_weights_critic_h5')

    def load_weights(self, path):
        self.model.load_weights(path)
