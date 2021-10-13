import numpy as np
import tensorflow as tf
import os
import sys
from six.moves import cPickle
import matplotlib.pyplot as plt
from tensorflow.python.keras import backend as K
from tensorflow.keras import models, layers
import pdb

path = "/home/zhangh/Documents/mypython/datasets/cifar-10-batches-py"


def resnet(input_image):
    x = layers.ZeroPadding2D((3,3))(input_image)
    x = layers.Conv2D(64, (7, 7), strides=(2, 2), name='conv1', use_bias=True)(x)
    x = layers.BatchNormalization(name='bn-conv1', trainable=True)(x)
    x = layers.Activation('relu')(x)
    C1 = x = layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
    return C1


def load_batch(fpath, label_key='labels'):
    """Internal utility for parsing CIFAR data.
    Arguments:
        fpath: path the file to parse.
        label_key: key for label data in the retrieve
            dictionary.
    Returns:
        A tuple `(data, labels)`.
    """
    with open(fpath, 'rb') as f:
        if sys.version_info < (3,):
            d = cPickle.load(f)
        else:
            d = cPickle.load(f, encoding='bytes')
        # decode utf8
        d_decoded = {}
        for k, v in d.items():
            d_decoded[k.decode('utf8')] = v
        d = d_decoded
    data = d['data']
    labels = d[label_key]

    data = data.reshape(data.shape[0], 3, 32, 32)
    return data, labels

if __name__ == "__main__":
    num_train_samples = 50000

    x_train = np.empty((num_train_samples, 3, 32, 32), dtype='uint8')
    y_train = np.empty((num_train_samples,), dtype='uint8')
    for i in range(1, 6):
        fpath = os.path.join(path, 'data_batch_' + str(i))
        (x_train[(i - 1) * 10000:i * 10000, :, :, :],
        y_train[(i - 1) * 10000:i * 10000]) = load_batch(fpath)

    fpath = os.path.join(path, 'test_batch')
    x_test, y_test = load_batch(fpath)

    y_train = np.reshape(y_train, (len(y_train), 1))
    y_test = np.reshape(y_test, (len(y_test), 1))

    if K.image_data_format() == 'channels_last':
        x_train = x_train.transpose(0, 2, 3, 1)
        x_test = x_test.transpose(0, 2, 3, 1)

    x_test = x_test.astype(x_train.dtype)
    y_test = y_test.astype(y_train.dtype)
    train_images, train_labels = x_train, y_train
    test_images, test_labels = x_test, y_test
    train_images = train_images / 255.
    test_images = test_images / 255.
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                'dog', 'frog', 'horse', 'ship', 'truck']

    model = models.Sequential()
    model.add(layers.Conv2D(32, 3, activation="relu", input_shape=(32,32,3)))
    model.add(layers.MaxPooling2D())
    model.add(layers.Conv2D(64, (3,3), activation="relu"))
    model.add(layers.MaxPooling2D())
    model.add(layers.Conv2D(64, (3,3), activation="relu"))
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(10))
    model.add(layers.Softmax())
    model.summary()

    model.compile(optimizer="adam",
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                  metrics=['accuracy']
    )
    # history = model.fit(train_images, train_labels, epochs=100,
    #                     steps_per_epoch=2000, validation_data=(test_images, test_labels))
    model1 = models.Sequential()
    C1 = resnet(train_images)
