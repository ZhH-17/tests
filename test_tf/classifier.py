''' test keras '''
import tensorflow as tf
import keras
import numpy as np
import datetime
import os

config = tf.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)

test = np.loadtxt("./iris_test.csv", skiprows=1, delimiter=",")
test_x, test_y = test[:, :-1], test[:, -1]
train = np.loadtxt("./iris_training.csv", skiprows=1, delimiter=",")
train_x, train_y = train[:, :-1], train[:, -1]

# model = tf.keras.Sequential([
#     KL.Dense(20, activation='relu', name="dense1"),
#     KL.Dense(3, name="dense2")])

inputs = keras.layers.Input(shape=(4,))
x = keras.layers.Dense(20, activation='relu', name="dense1")(inputs)
outputs = keras.layers.Dense(3, name="dense2")(x)
model = keras.models.Model(inputs=inputs, outputs=outputs)
# model.compile(optimizer="Adam", loss="mse", metrics=["mae"])


now = datetime.datetime.now()
log_dir = "log_{:%Y%m%dT%H%M}".format(now)
if not os.path.isdir(log_dir):
    os.mkdir(log_dir)
checkpoint_path = os.path.join(log_dir, "iris_{epoch:04d}.h5")

callbacks = [keras.callbacks.ModelCheckpoint(checkpoint_path, verbose=0,
                                                save_weights_only=True, period=2)]
model.compile(optimizer='adam',
              # loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              loss=keras.losses.sparse_categorical_crossentropy,
              metrics=['accuracy'])
model.fit(train_x, train_y, epochs=20,
            callbacks=callbacks)

model.save_weights("weights.h5")
test_loss, test_acc = model.evaluate(test_x,  test_y, verbose=2)
print('\nTest accuracy:', test_acc)
