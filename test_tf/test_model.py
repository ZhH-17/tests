import numpy as np
import tensorflow as tf
import os
import sys
from six.moves import cPickle
from tensorflow.python.keras import backend as K
from tensorflow.keras import models, layers
from tensorflow import keras

class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        self.conv1 = Conv2D(32, 3, activation='relu')
        self.flatten = Flatten()
        self.d1 = Dense(128, activation='relu')
        self.d2 = Dense(10, activation='softmax')

    def call(self, x):
        x = self.conv1(x)
        x = self.flatten(x)
        x = self.d1(x)
        return self.d2(x)


class identity(tf.keras.Model):
    def __init__(self):
        super(identity, self).__init__()
        self.conv1 = layers.Conv2D(10, (3, 3), name='conv1')
        self.conv2 = layers.Conv2D(10, (3, 3), name='')

    def call(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x

class conv_block(keras.Model)
    def __init__(self):
        super(identity, self).__init__()
        self.conv1 = layers.Conv2D(10, (3, 3), name='conv1')
        self.conv2 = layers.Conv2D(10, (3, 3), name='')


if __name__ == "__main__":
    model = keras.Sequential()
    # model.add(layers.Conv2D(32, 3, activation="relu", input_shape=(1,2,1)))
    model.add(layers.Dense(10, activation='relu', name='den_1', input_shape=[3], use_bias=False))
    model.add(layers.Dense(20, activation='relu', name='den_2'))
    # model.add(layers.Dense(5, activation='relu', name='den_3'))
    # model.add(layers.Flatten())
    model.add(layers.Dense(5, name='den_4'))
    model.add(layers.Activation(activation='softmax'))
    model.summary()
    opti = keras.optimizers.Adam()

    model.compile(opti, loss=keras.losses.sparse_categorical_crossentropy, metrics=['accuracy'])

    x = tf.random.normal((10, 3))
    y = 3*x[:, 0] + 2*x[:, 1] + 5 * x[:, 2]
    y = np.zeros(len(x))
    for i in range(len(x)):
      xi = x[i]
      if xi[0] >= 0 and xi[1] >= 0:
        y[i] = 1
      elif xi[0] < 0 and xi[1] >= 0:
        y[i] = 2
      elif xi[0] < 0 and xi[1] < 0:
        y[i] = 3
      else:
        y[i] = 4
    history = model.fit(x, y, epochs=10)
    model.predict(x[:2])


    # model = MyModel()

    # with tf.GradientTape() as tape:
    # logits = model(images)
    # loss_value = loss(logits, labels)
    # grads = tape.gradient(loss_value, model.trainable_variables)
    # optimizer.apply_gradients(zip(grads, model.trainable_variables))
