import tensorflow as tf
import tensorflow.keras.layers as KL
import tensorflow_datasets as tfds
import numpy as np
from datetime import datetime
import pdb

def correct_pad(input_size, kernel_size):
    '''
    fit stride=2, correct image size
    '''
    correct = (kernel_size[0] // 2, kernel_size[1] // 2)
    # even:1, odd: 0
    adjust = (1 - input_size[0] % 2, 1 - input_size[1] % 2)
    return ((correct[0] - adjust[0], correct[0]),
            (correct[1] - adjust[1], correct[1]))


class unetModel(tf.keras.Model):
    def __init__(self):
        super(unetModel, self).__init__()
        self.conv_l1 = tf.keras.layers.Conv2D(64, 3, padding='valid', activation='relu')
        self.conv_l2 = tf.keras.layers.Conv2D(128, 3, padding='valid', activation='relu')
        self.conv_l3 = tf.keras.layers.Conv2D(256, 3, padding='valid', activation='relu')
        self.conv_l4 = tf.keras.layers.Conv2D(512, 3, padding='valid', activation='relu')

        self.conv_s2 = tf.keras.layers.Conv2D(256, 3, padding='valid', strides=(2, 2),
                                              activation='softmax')
        self.flatten = tf.keras.layers.Flatten()
        self.dense = tf.keras.layers.Dense(10)


    def call(self, x):
        x = self.conv_l1(x)
        x = self.flatten(x)
        x = self.dense(x)

        return x

def _inverted_res_block(inputs, expansion, stride, alpha, filters, block_id):
    in_channels = inputs.shape[-1]
    adjust_filters = int(filters * alpha)
    pw_filters = adjust_filters // 8 * 8

    if block_id:
        # expand
        pass


def depthwise_conv_block(inputs, pointwise_filters, alpha, strides, block_id):
    if strides == (1, 1):
        x = inputs
    else:
        x = KL.ZeroPadding2D(((0, 1), (0, 1)), name='conv_pad_%d' % block_id)(inputs)
    x = KL.DepthwiseConv2D(kernel_size=3,
                           padding='same' if strides == (1, 1) else 'valid',
                           strides=strides,
                           use_bias=False,
                           name="conv_dw_%d" % block_id)(x)
    pointwise_filters = int(alpha * pointwise_filters)

    x = KL.BatchNormalization(name='conv_dw_%d_bn' % block_id)(x)
    x = KL.ReLU(6., name="conv_dw_%d_relu" % block_id)(x)
    x = KL.Conv2D(pointwise_filters,
                  kernel_size=1,
                  padding='same',
                  use_bias=False,
                  strides=(1, 1),
                  name="conv_pw_%d" % block_id)(x)
    x = KL.BatchNormalization(name='conv_pw_%d_bn' % block_id)(x)
    x = KL.ReLU(6., name="conv_pw_%d_relu" % block_id)(x)
    return x


def mobileNet(input_shape, classes, alpha=1., dropout=0):
    inputs = KL.Input(shape=input_shape)
    x = inputs
    x = tf.keras.layers.Conv2D(32, 3, strides=(2, 2),
                               padding='same', use_bias=False, name='Conv1')(x)
    x = tf.keras.layers.BatchNormalization(
        epsilon=1e-3, momentum=0.999, name='bn_Conv1')(x)
    x = KL.ReLU(max_value=6.)(x)
    alpha = 1.
    x = depthwise_conv_block(x, 64, alpha, (1, 1), 0)

    x = depthwise_conv_block(x, 128, alpha, (2, 2), 1)
    x = depthwise_conv_block(x, 128, alpha, (1, 1), 2)

    x = depthwise_conv_block(x, 256, alpha, (2, 2), 3)
    x = depthwise_conv_block(x, 256, alpha, (1, 1), 4)

    x = depthwise_conv_block(x, 512, alpha, (2, 2), 5)

    x = depthwise_conv_block(x, 512, alpha, (1, 1), 6)
    x = depthwise_conv_block(x, 512, alpha, (1, 1), 7)
    x = depthwise_conv_block(x, 512, alpha, (1, 1), 8)
    x = depthwise_conv_block(x, 512, alpha, (1, 1), 9)
    x = depthwise_conv_block(x, 512, alpha, (1, 1), 10)

    x = depthwise_conv_block(x, 1024, alpha, (2, 2), 11)
    x = depthwise_conv_block(x, 1024, alpha, (2, 2), 12)

    x = KL.GlobalAveragePooling2D()(x)
    x = KL.Reshape((1, 1, int(alpha * 1024)), name='reshape_1')(x)
    x = KL.Dropout(dropout, name='dropout')(x)
    x = KL.Conv2D(classes, (1, 1), padding='same', name='conv_pred')(x)
    x = KL.Reshape((classes,), name='reshape_2')(x)
    x = KL.Activation("softmax", name='act_softmax')(x)

    rows = inputs.shape[1]
    model = tf.keras.Model(inputs, x, name='mobilnet_%0.2f_%s' %(alpha, rows))
    return model

class InstanceNormalization(tf.keras.layers.Layer):
  """Instance Normalization Layer (https://arxiv.org/abs/1607.08022)."""

  def __init__(self, epsilon=1e-5):
    super(InstanceNormalization, self).__init__()
    self.epsilon = epsilon

  def build(self, input_shape):
    self.scale = self.add_weight(
        name='scale',
        shape=input_shape[-1:],
        initializer=tf.random_normal_initializer(1., 0.02),
        trainable=True)

    self.offset = self.add_weight(
        name='offset',
        shape=input_shape[-1:],
        initializer='zeros',
        trainable=True)

  def call(self, x):
    mean, variance = tf.nn.moments(x, axes=[1, 2], keepdims=True)
    inv = tf.math.rsqrt(variance + self.epsilon)
    normalized = (x - mean) * inv
    return self.scale * normalized + self.offset


def downsample(filters, size, norm_type='batchnorm', apply_norm=True):
  """Downsamples an input.
  Conv2D => Batchnorm => LeakyRelu
  Args:
    filters: number of filters
    size: filter size
    norm_type: Normalization type; either 'batchnorm' or 'instancenorm'.
    apply_norm: If True, adds the batchnorm layer
  Returns:
    Downsample Sequential Model
  """
  initializer = tf.random_normal_initializer(0., 0.02)

  result = tf.keras.Sequential()
  result.add(
      tf.keras.layers.Conv2D(filters, size, strides=2, padding='same',
                             kernel_initializer=initializer, use_bias=False))

  if apply_norm:
    if norm_type.lower() == 'batchnorm':
      result.add(tf.keras.layers.BatchNormalization())
    elif norm_type.lower() == 'instancenorm':
      result.add(InstanceNormalization())

  result.add(tf.keras.layers.LeakyReLU())

  return result


def upsample(filters, size, norm_type='batchnorm', apply_dropout=False):
  """Upsamples an input.
  Conv2DTranspose => Batchnorm => Dropout => Relu
  Args:
    filters: number of filters
    size: filter size
    norm_type: Normalization type; either 'batchnorm' or 'instancenorm'.
    apply_dropout: If True, adds the dropout layer
  Returns:
    Upsample Sequential Model
  """

  initializer = tf.random_normal_initializer(0., 0.02)

  result = tf.keras.Sequential()
  result.add(
      tf.keras.layers.Conv2DTranspose(filters, size, strides=2,
                                      padding='same',
                                      kernel_initializer=initializer,
                                      use_bias=False))

  if norm_type.lower() == 'batchnorm':
    result.add(tf.keras.layers.BatchNormalization())
  elif norm_type.lower() == 'instancenorm':
    result.add(InstanceNormalization())

  if apply_dropout:
    result.add(tf.keras.layers.Dropout(0.5))

  result.add(tf.keras.layers.ReLU())

  return result

base_model = tf.keras.applications.MobileNetV2(input_shape=(128, 128, 3), include_top=False)
# Use the activations of these layers
layer_names = [
    'block_1_expand_relu',   # 64x64
    'block_3_expand_relu',   # 32x32
    'block_6_expand_relu',   # 16x16
    'block_13_expand_relu',  # 8x8
    'block_16_project',      # 4x4
]
base_model_outputs = [base_model.get_layer(name).output for name in layer_names]

down_stack = tf.keras.Model(inputs=base_model.input, outputs=base_model_outputs)

down_stack.trainable = False

up_stack = [
    upsample(512, 3),  # 4x4 -> 8x8
    upsample(256, 3),  # 8x8 -> 16x16
    upsample(128, 3),  # 16x16 -> 32x32
    upsample(64, 3),   # 32x32 -> 64x64
]


def unet_model(output_channels: int):
    inputs = KL.Input(shape=(128, 128, 3))
    skips = down_stack(inputs)
    x =  skips[-1]
    skips = reversed(skips[:-1])

    for up, skip in zip(up_stack, skips):
        x = up(x)
        concat = KL.Concatenate()
        x = concat([x, skip])

    last = tf.keras.layers.Convolution2DTranspose(
        filters=output_channels,
        kernel_size=3,
        strides=2,
        padding='same'
    )
    x = last(x)

    return tf.keras.Model(inputs=inputs, outputs=x)


def create_mask(pred_mask):
  pred_mask = tf.argmax(pred_mask, axis=-1)
  pred_mask = pred_mask[..., tf.newaxis]
  return pred_mask[0]


if __name__ == "__main__":
    OUTPUT_CLASSES = 3

    model = unet_model(output_channels=OUTPUT_CLASSES)

    # dataset, info = tfds.load('oxford_iiit_pet:3.*.*', with_info=True)
    # model = mobileNet((224, 224, 3), 10)
    # model = unetModel()

    # fashion_mnist = tf.keras.datasets.fashion_mnist
    # (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
    # train_images = np.expand_dims(train_images, axis=3) / 255.
    # test_images = np.expand_dims(test_images, axis=3) / 255.
    # class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
    #             'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

    # loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    # optimizer = tf.keras.optimizers.Adam()

    # train_loss = tf.keras.metrics.Mean(name='train_loss')
    # train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

    # test_loss = tf.keras.metrics.Mean(name='test_loss')
    # test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')

    # log_dir = "./logs/" + datetime.now().strftime("%Y%m%d-%H%M%S")
    # tb_cb = tf.keras.callbacks.TensorBoard(log_dir=log_dir)
    # model.compile(optimizer,
    #             loss=loss_object,
    #             metrics=[train_accuracy])
    # model.fit(train_images,
    #         train_labels,
    #         batch_size=50,
    #         epochs=5,
    #         callbacks=[tb_cb]
    #         )


