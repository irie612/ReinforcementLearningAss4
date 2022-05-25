import numpy as np
import os
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
import tensorflow_probability as tfp

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}

initializer = tf.keras.initializers.Zeros()


class MonteCarloG:

    def __init__(self, state_shape, n_actions, learning_rate, gamma):
        self.state_shape = state_shape
        self.n_actions = n_actions
        self.learning_rate = learning_rate
        self.gamma = gamma

        # check model, maybe get it out of class ?
        self.model = Sequential()
        self.model.add(Conv2D(32, (3, 3), activation='relu', input_shape=state_shape))
        self.model.add(MaxPooling2D((2, 2)))
        self.model.add(Conv2D(64, (3, 3), activation='relu'))
        self.model.add(MaxPooling2D((2, 2)))
        self.model.add(Conv2D(64, (3, 3), activation='relu'))
        self.model.add(Flatten())
        self.model.add(Dense(64, activation='relu', kernel_initializer=initializer))
        self.model.add(Dense(n_actions, activation='softmax'))
        self.model.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate))

    def select_action(self, s):
        probs = self.model.predict(np.array([s]), verbose=0)[0]
        # print(probs)
        a = np.random.choice(self.n_actions, p=probs)
        return a, probs

    def update(self, r, a, s):
        G = 0
        G_list = []
        for t in range(len(r) - 1, -1, -1):
            G += r[t]
            G_list.append(G)
            with tf.GradientTape() as tape:
                p = self.model(np.array([s[t]]), training=True)
                loss = G * tfp.distributions.Categorical(probs=p[0]).log_prob(a[t])
            grads = tape.gradient(loss, self.model.trainable_variables)
            self.model.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
