import os
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
import tensorflow_probability as tfp
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}


def get_model(input_shape, output, lr):
    model = Sequential()
    model.add(tf.keras.Input(shape=input_shape))
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
        self.model = get_model(state_shape, n_actions, learning_rate)

    def select_action(self, s):
        probs = self.model.predict(np.array([s]), verbose=0)[0]
        a = np.random.choice(self.n_actions, p=probs)
        return a, probs

    def update(self, r, a, s):
        G = 0
        G_list = []
        for t in range(len(r) - 2, -1, -1):
            G = r[t+1] + G * self.gamma
            G_list.insert(0, G)
        for t in range(len(r)-1):
            with tf.GradientTape() as tape:
                p = self.model(np.array([s[t]]))
                loss = -self.gamma**t * G_list[t] * tfp.distributions.Categorical(probs=p[0]).log_prob(a[t])
            grads = tape.gradient(loss, self.model.trainable_variables)
            self.model.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

    def save_model(self, name):
        self.model.save(f'saved_models/{name}')
