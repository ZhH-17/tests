import tensorflow as tf
import numpy as np

img = np.random.randint(0, 255, (100, 100, 3))

inputs = tf.keras.layers.Input(shape=[100, 100, 3])
x = inputs
outputs = tf.keras.layers.Conv2D(10, 3)(x)
model = tf.keras.Model(inputs=inputs, outputs=outputs)
