import numpy as np
import os
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
import tensorflow_probability as tfp
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}


def get_model(input, output, lr):
    model = Sequential()
    model.add(tf.keras.Input(shape=input))
    model.add(Dense(12, activation='relu'))
    model.add(Dense(24, activation='relu'))
    model.add(Dense(12, activation='relu'))
    model.add(Dense(output, activation='softmax'))
    model.compile(optimizer=tf.keras.optimizers.Adam(lr))
    return model

class MonteCarloG:

    def __init__(self, state_shape, n_actions, learning_rate, gamma):
        self.state_shape = state_shape
        self.n_actions = n_actions
        self.learning_rate = learning_rate
        self.gamma = gamma

        # check model, maybe get it out of class ?
        self.model = get_model(state_shape, n_actions, learning_rate)
        
        # self.model.add(Conv2D(32, (3, 3), activation='relu', input_shape=state_shape))
        # self.model.add(MaxPooling2D((2, 2)))
        # self.model.add(Conv2D(64, (3, 3), activation='relu'))
        # self.model.add(MaxPooling2D((2, 2)))
        # self.model.add(Conv2D(64, (3, 3), activation='relu'))
        # self.model.add(Flatten())
        # self.model.add(Dense(64, activation='relu', kernel_initializer=initializer))
        # self.model.add(Dense(32, activation='relu', kernel_initializer=initializer))
        # self.model.add(Dense(n_actions, activation='softmax'))
        # self.model.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate))

    def select_action(self, s):
        probs = self.model.predict(np.array([s]), verbose=0)[0]#, verbose=0)[0]
        # print(probs)
        a = np.random.choice(self.n_actions, p=probs)
        return a, probs

    def update(self, r, a, s):
        G = 0
        G_list = []
        # for t in range(0, len(r)):
        for t in range(len(r) - 2, -1, -1):
            G = r[t+1] + G * self.gamma
            G_list.insert(0,G)
            # with tf.GradientTape() as tape:
            #     p = self.model(np.array([s[t]]), training=True)
            #     loss = -self.gamma**t * G * tfp.distributions.Categorical(probs=p[0]).log_prob(a[t])
            # grads = tape.gradient(loss, self.model.trainable_variables)
            # self.model.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        #print(G_list)
        #G_list = np.array(G_list)
        #G_list = ( G_list - np.std(G_list) ) / np.mean(G_list)
        for t in range(len(r)-1):
            with tf.GradientTape() as tape:
                p = self.model(np.array([s[t]])) 
                #if t==1:
                    #print(s[t],a[t], tfp.distributions.Categorical(probs=p[0]).log_prob(a[t]))
                loss = -self.gamma**t * G_list[t] * tfp.distributions.Categorical(probs=p[0]).log_prob(a[t])
            grads = tape.gradient(loss, self.model.trainable_variables)
            # if t==5:
            #     print(grads)
            self.model.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
            
