import os
import pathlib
import time
import tensorflow as tf
from tensorflow.keras import models, Model, layers
from tensorflow.keras import backend
import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv
import pdb

AUTOTUNE = tf.data.experimental.AUTOTUNE
BATCH_SIZE = 32

data_dir = "/home/zhangh/Documents/mypython/datasets/flower_photos/"
data_dir = pathlib.Path(data_dir)
image_count = len(list(data_dir.glob('*/*.jpg')))
print(image_count)

batch_size = 32
img_height = 180
img_width = 180

def process_image(image):
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, (192, 192))
    image = image / 255.
    return image

def correct_pad(backend, inputs, kernel_size):
    """Returns a tuple for zero-padding for 2D convolution with downsampling.

    # Arguments
        input_size: An integer or tuple/list of 2 integers.
        kernel_size: An integer or tuple/list of 2 integers.

    # Returns
        A tuple.
    """
    img_dim = 2 if backend.image_data_format() == 'channels_first' else 1
    input_size = backend.int_shape(inputs)[img_dim:(img_dim + 2)]

    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)

    if input_size[0] is None:
        adjust = (1, 1)
    else:
        adjust = (1 - input_size[0] % 2, 1 - input_size[1] % 2)

    correct = (kernel_size[0] // 2, kernel_size[1] // 2)

    return ((correct[0] - adjust[0], correct[0]),
            (correct[1] - adjust[1], correct[1]))

def _make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v

def bottleneck(x, filters, alpha, stride, t, block_id):
    channel_axis = 1 if backend.image_data_format() == 'channels_first' else -1

    in_channels = backend.int_shape(x)[channel_axis]
    pointwise_conv_filters = int(filters * alpha)
    pointwise_filters = _make_divisible(pointwise_conv_filters, 8)

    prefix = "block_{}_".format(block_id)
    inputs = x

    x = layers.Conv2D(t*in_channels, kernel_size=1,
                      padding="same", use_bias=False,
                      activation=None, name=prefix+"expand")(x)
    x = layers.BatchNormalization(axis=channel_axis, epsilon=1.e-3,
                                  momentum=0.999, name=prefix+"expand_BN")(x)
    x = layers.ReLU(6, name=prefix+"expand_relu")(x)

    if stride == 2:
        x = layers.ZeroPadding2D(padding=correct_pad(backend, x, 3), name=prefix+'pad')(x)
    x = layers.DepthwiseConv2D(kernel_size=3,
                               strides=stride,
                               activation=None,
                               use_bias=False,
                               padding='same' if stride == 1 else 'valid',
                               name=prefix + 'depthwise')(x)
    x = layers.BatchNormalization(axis=channel_axis, epsilon=1.e-3,
                                  momentum=0.999, name=prefix+"depthwise_BN")(x)
    x = layers.ReLU(6, name=prefix+"depthwise_relu")(x)

    x = layers.Conv2D(pointwise_filters, 1, padding='same',
                      use_bias=False, activation=None, name=prefix+"project")(x)
    x = layers.BatchNormalization(axis=channel_axis,
                                  epsilon=1e-3,
                                  momentum=0.999,
                                  name=prefix + 'project_BN')(x)

    if in_channels == pointwise_filters and stride == 1:
        return layers.Add(name=prefix + 'add')([inputs, x])
    return x


def mobileNet(input_shape=None, alpha=1.0, include_top=True, weights='imagenet',
              input_tensor=None, pooling=None, classes=1000):
    model = tf.keras.Sequential()

    rows = input_shape[0]
    cols = input_shape[1]

    if rows == cols and rows in [96, 128, 160, 192, 224]:
        default_size = rows
    else:
        default_size = 224

    if include_top:
        input_shape =(default_size, default_size, 3)
    else:
        input_shape =(None, None, 3)

    if input_tensor is None:
        img_input = layers.Input(shape=input_shape)
    else:
        if not backend.is_keras_tensor(input_tensor):
            img_input = layers.Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor

    first_block_filters = _make_divisible(32 * alpha, 8)
    x = layers.ZeroPadding2D(padding=correct_pad(backend, img_input, 3),
                             name="conv1_pad")(img_input)
    x = layers.Conv2D(first_block_filters, kernel_size=3,
                      strides=(2, 2), padding='valid',
                      use_bias=False, name="conv1")(x)
    x = layers.BatchNormalization(axis=-1, epsilon=1.e-3,
                                  momentum=0.999,
                                  name="bn_conv1")(x)
    # 1st type
    x = bottleneck(x, 16, alpha, stride=1, t=1, block_id=0)

    # 2nd type
    x = bottleneck(x, 24, alpha, stride=2, t=6, block_id=1)
    x = bottleneck(x, 24, alpha, stride=1, t=6, block_id=2)

    # 3rd type
    x = bottleneck(x, 32, alpha, stride=2, t=6, block_id=3)
    x = bottleneck(x, 32, alpha, stride=1, t=6, block_id=4)
    x = bottleneck(x, 32, alpha, stride=1, t=6, block_id=5)

    # 4th type
    x = bottleneck(x, 64, alpha, stride=2, t=6, block_id=6)
    x = bottleneck(x, 64, alpha, stride=1, t=6, block_id=7)
    x = bottleneck(x, 64, alpha, stride=1, t=6, block_id=8)
    x = bottleneck(x, 64, alpha, stride=1, t=6, block_id=9)

    # 5th type
    x = bottleneck(x, 96, alpha, stride=1, t=6, block_id=10)
    x = bottleneck(x, 96, alpha, stride=1, t=6, block_id=11)
    x = bottleneck(x, 96, alpha, stride=1, t=6, block_id=12)

    # 6th type
    x = bottleneck(x, 160, alpha, stride=2, t=6, block_id=13)
    x = bottleneck(x, 160, alpha, stride=1, t=6, block_id=14)
    x = bottleneck(x, 160, alpha, stride=1, t=6, block_id=15)

    # 7th
    x = bottleneck(x, 320, alpha, stride=1, t=6, block_id=13)

    x = layers.Conv2D(1280, 1, padding='same', use_bias=False, name="conv_1")(x)
    x = layers.BatchNormalization(axis=-1, epsilon=1.e-3,
                                  momentum=0.999, name="conv_1_bn")(x)
    x = layers.ReLU(6., name='out_relu')(x)

    if include_top:
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dense(classes, activation='softmax',
                         use_bias=True, name='Logits')(x)
    else:
        if pooling == 'avg':
            x = layers.GlobalAveragePooling2D()(x)
        elif pooling == 'max':
            x = layers.GlobalMaxPooling2D()(x)

    if input_tensor is not None:
        inputs = keras_utils.get_source_inputs(input_tensor)
    else:
        inputs = img_input
    model = models.Model(inputs, x, name="mobilenet_%0.2f_%s" %(alpha, default_size))

def load_and_preprocess_image(fn):
    image = tf.io.read_file(fn)
    return process_image(image)


def image_dataset_from_directory(data_dir, validation_split=None):
    label_names = sorted(item.name for item in data_dir.glob('*/') if item.is_dir())
    label_to_index = dict((name, index) for index, name in enumerate(label_names))
    all_image_paths = []
    for label in label_names:
        fns = list(data_dir.glob("%s/*.jpg" %label))
        all_image_paths.extend( list(map(str, fns)) )
    np.random.shuffle(all_image_paths)

    all_image_labels = []
    for fn in all_image_paths:
        name = os.path.basename(os.path.dirname(fn))
        all_image_labels.append(label_to_index[name])
    if validation_split is not None:
        val_num = int(len(all_image_paths) * validation_split)
        all_image_paths_val = all_image_paths[:val_num]
        all_image_labels_val = all_image_labels[:val_num]

    path_ds = tf.data.Dataset.from_tensor_slices(all_image_paths)
    image_ds = path_ds.map(load_and_preprocess_image, num_parallel_calls=AUTOTUNE)
    label_ds = tf.data.Dataset.from_tensor_slices(
        tf.cast(all_image_labels, tf.int64))

    image_label_ds = tf.data.Dataset.zip((image_ds, label_ds))
    return image_label_ds


label_names = sorted(item.name for item in data_dir.glob('*/') if item.is_dir())
train_ds = image_dataset_from_directory(data_dir)
val_ds = image_dataset_from_directory(data_dir, validation_split=0.8)
val_ds = val_ds.cache().batch(BATCH_SIZE).prefetch(buffer_size=AUTOTUNE)

# 设置一个和数据集大小一致的 shuffle buffer size（随机缓冲区大小）以保证数据被充分打乱。
ds = train_ds.apply(tf.data.experimental.shuffle_and_repeat(buffer_size=image_count))
ds = ds.batch(BATCH_SIZE)
ds = ds.prefetch(buffer_size=AUTOTUNE)

# model_net = tf.keras.applications.MobileNetV2(input_shape=(192,192,3), include_top=False)
model_net = mobileNet(input_shape=(192,192,3), include_top=False)
model_net.trainable = False

def change_range(image, label):
    return image*2 - 1, label

keras_ds = ds.map(change_range)
model = tf.keras.Sequential([model_net,
                             layers.GlobalAveragePooling2D(),
                             layers.Dense(len(label_names), activation="softmax")])

# feature_map_batch = model(image_batch)
# print(feature_map_batch.shape)

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=["accuracy"])
history = model.fit(keras_ds,
                    validation_data=val_ds,
                    epochs=10, steps_per_epoch=30)

acc = history.history['acc']
val_acc = history.history['val_acc']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()
