import tensorflow as tf
import keras.backend as K
from keras.initializers import RandomUniform
from keras.layers import Dense, Flatten, Input, merge, Lambda, Activation, ThresholdedReLU, BatchNormalization, concatenate, Multiply, Subtract
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras.callbacks import History
from keras import optimizers
import random
import numpy as np
from keras.regularizers import l1_l2
from itertools import product
history = History()

from keras.utils.generic_utils import get_custom_objects


class ReLUs(Activation):
    def __init__(self, activation, **kwargs):
        super(ReLUs, self).__init__(activation, **kwargs)
        self.__name__ = 'ReLU_s'

# Deep Deterministc Policy Gradient Agent
class Actor:
    def __init__(self, state_size=28, action_size=2, learning_rate=0.001, tau=0.001, sess=None, batch_size = 32, action_max=[1000, 2]):
        self.state_size = state_size        # state size
        self.action_size = action_size      # action size
        self.learning_rate = learning_rate  # learning rate for actor
        self.tau = tau                      # soft copy factor
        self.batch_size = batch_size        # batch size
        self.action_max_range = action_max  # max value for action
        #self.act_range = act_range
        # self.model = self._build_actor_model()
        # self.target_model = self._build_actor_model()
        #self.adam_optimizer = self.optimizer()
        self.model, self.weights, self.input_states = self.create_network()   # create actor network
        self.target_model, self.target_action, self.target_state = self.create_network()    # create target network
        # self.adam_optimizer = self.optimizer()      # init adam optimizer
        self.sess = sess
        self.action_gradient = tf.placeholder(tf.float32, [None, self.action_size])
        self.params_grad = tf.gradients(self.model.output, self.model.trainable_weights, -self.action_gradient)
        self.grads = zip(self.params_grad, self.model.trainable_weights)
        self.optimize = tf.train.AdamOptimizer(self.learning_rate).apply_gradients(self.grads)
        self.sess.run(tf.initialize_all_variables())

    def create_network(self):
        '''
        Create network
        :return: returns model, trainable weights and Input
        '''
        S = Input(shape=[self.state_size])      # input layer of state size
        b1 = BatchNormalization()(S)            # batch normalization
        # h0 = Dense(32, activation='sigmoid', kernel_initializer=RandomUniform())(b1)
        # h1 = Dense(32, activation='linear')(b1)  # 64     300   128
        # h2 = Dense(32, activation='relu')(h1)  # 64     300     128
        # h3 = Dense(32, activation='relu')(h2)  # 64     300    128
        h4 = Dense(3* self.state_size, activation='relu')(b1)  # 64     300   128           # hidden layers
        h5 = Dense(3* self.state_size, activation='relu')(h4)  # 64     300     128
        h6 = Dense(3* self.state_size, activation='relu')(h5)  # 64     300    128
        h7 = Dense(5* self.state_size, activation='relu')(h6)  # 64     300    128
        # h1 = Dense(32, activation='linear')(b1)  # 64     300   128
        # h2 = Dense(32, activation='relu')(h1)  # 64     300     128
        # h3 = Dense(32, activation='relu')(h2)  # 64     300    128
        # h4 = Dense(32, activation='relu')(h3)  # 64     300    128
        # h5 = Dense(32, activation='relu')(h4)  # 64     300    128
        # h6 = Dense(32, activation='relu')(h5)  # 64     300    128
        # h7 = Dense(32, activation='relu')(h6)  # 64     300    128
        # h8 = Dense(64, activation='relu')(h3)  #64     300    128
        # h8 = Dense(64, activation='relu')(h3)
        # out = Dense(self.action_size, activation='relu', kernel_initializer=RandomUniform())(h5)  # 300    128
        out = Dense(self.action_size, activation=Activation(lambda x: K.relu(x, alpha=[0]*self.action_size, max_value=[1]*self.action_size)), kernel_initializer=RandomUniform(seed=2))(h7) # output layer

        # wl = Dense(self.action_size-1, activation='softmax')(h3)
        # blkl = Dense(1, activation='relu')(h3)
        # out = concatenate([blkl, wl])

        # h6 = Dense(64, activation='relu')(h5)
        # h6 = Dense(64, activation='relu')(h5)
        # h4 = Dense(5*self.state_size, activation='relu')(h3)
        # b4 = BatchNormalization()(h4)

        # h2 = Dense(32, activation='relu')(h0)      #300     128
        # h3 = Dense(32, activation='relu')(h2)       #300    128
        # h4 = Dense(64, activation='relu')(h3)       #300    128
        # h3 = Dense(64, activation='relu')(h2)
        # b3 = BatchNormalization()(h1)
        # h3 = Dense(2, activation=Activation(lambda x: K.relu(x, alpha=[0.0, 0.0], max_value=self.action_max_range)))(h2))
        # out = Dense(self.action_size, activation=Activation(lambda x: K.relu(x, alpha=[0]*self.action_size, max_value=[1]*self.action_size)), kernel_initializer=RandomUniform(seed=2))(h4)

        # b3_1 = Dense(1, activation='sigmoid')(h3)
        # b3_2 = Dense(1, activation='sigmoid')(h3)
        # b3_3_in = concatenate([b3_2, h3])
        # b3_3 = Dense(1, activation='sigmoid')(b3_3_in)
        # # b3_4_in = concatenate([b3_3, h3])
        # # b3_4 = Dense(1, activation='sigmoid')(b3_4_in)
        # out = concatenate([b3_1, b3_2, b3_3])
        # a1 = Dense(1, activation=Activation(lambda x: K.relu(x, alpha=0, max_value=1)),kernel_initializer=RandomUniform())(h2)
        # a2 = Dense(1, activation=Activation(lambda x: K.relu(x, alpha=0, max_value=1)),kernel_initializer=RandomUniform())(h2)
        # a3 = Dense(1, activation=Activation(lambda x: K.relu(x, alpha=0, max_value=1)),kernel_initializer=RandomUniform())(h2)
        # from_a3 = Multiply()([Lambda(lambda  x: 1-x)(a2), a3])
        # from_a4 = Subtract()([Lambda(lambda  x: 1-x)(a2), from_a3])
        # out = concatenate([a1, a2, from_a3, from_a4])

        # out_b = BatchNormalization()(h4)
        # out = Dense(self.action_size, activation='relu', kernel_initializer=RandomUniform())(out)
        #out = BatchNormalization()(out)

        model = Model(input=S, output=out)          # create model
        return model, model.trainable_weights, S        # return model and trainable weights

    def transfer_to_actor_model(self):
        '''
        Soft copy to target network with factor tau
        :return:
        '''
        W, target_W = self.model.get_weights(), self.target_model.get_weights()
        for i in range(len(W)):
            target_W[i] = self.tau * W[i] + (1 - self.tau) * target_W[i]
        self.target_model.set_weights(target_W)

    def predict(self, state):
        '''
        decide on actions
        :param state:
        :return:
        '''
        return self.model.predict(state)

    def target_predict(self, input):
        '''
        target network decides on action
        :param input:
        :return:
        '''
        #print('act target predict')
        input = np.asarray(input).reshape((self.batch_size, self.state_size))
        #print(input)
        return self.target_model.predict(input)

    # def train(self, states, grads):
    #     self.adam_optimizer([states, grads])
    #     # print('train')
    #     # print(p)
    #     # print('hier')
    #     # print(self.model.get_weights())

    def train_2(self, states, action_grads):
        '''
        Train based on states and action gradient
        :param states:
        :param action_grads:
        :return:
        '''
        self.sess.run(self.optimize, feed_dict={
        self.input_states: states,
        self.action_gradient: action_grads
        })
        # print('hier')
        # print(self.model.get_weights())
        #print(result)

    # def optimizer(self):
    #     action_gdts = K.placeholder(shape=(None, self.action_size))
    #     params_grad = tf.gradients(self.model.output, self.model.trainable_weights, -action_gdts)
    #     grads = zip(params_grad, self.model.trainable_weights)
    #     return K.function([self.model.input, action_gdts], [tf.train.AdamOptimizer(self.learning_rate).apply_gradients(grads)])

    def save(self, path):
        '''
        save weights of actor in path
        :param path:
        '''
        self.model.save_weights(path + '_weights_actor_h5')

    def load_weigths(self, path):
        '''
        load weights of actor in given path
        :param path:
        :return:
        '''
        self.model.load_weights(path)
