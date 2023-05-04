# CycleGAN generator and discriminator
# Author: Ruochong Wu

import tensorflow as tf
import tensorflow_addons as tfa


def residual_block(x):
    """Residual block"""
    initializer = tf.random_normal_initializer(0., 0.02)
    dim = x.shape[-1]
    input_tensor = x
    x = tf.keras.layers.Conv2D(dim, 3, strides=1, padding='same', kernel_initializer=initializer, use_bias=False)(x)
    x = tfa.layers.InstanceNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Conv2D(dim, 3, strides=1, padding='same', kernel_initializer=initializer, use_bias=False)(x)
    x = tfa.layers.InstanceNormalization()(x)
    x = tf.keras.layers.Add()([input_tensor, x])
    return x


def generator(output_channels=3):
    """ResNet generator, following architecture defined in report"""
    initializer = tf.random_normal_initializer(0., 0.02)
    inputs = tf.keras.layers.Input(shape=[256, 256, 3])
    x = inputs

    # Downsampler
    # First layer
    x = tf.keras.layers.Conv2D(64, 7, strides=1, padding='same', kernel_initializer=initializer, use_bias=False)(x)
    x = tfa.layers.InstanceNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    # Second layer
    x = tf.keras.layers.Conv2D(128, 3, strides=2, padding='same', kernel_initializer=initializer, use_bias=False)(x)
    x = tfa.layers.InstanceNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    # Third layer
    x = tf.keras.layers.Conv2D(256, 3, strides=2, padding='same', kernel_initializer=initializer, use_bias=False)(x)
    x = tfa.layers.InstanceNormalization()(x)
    x = tf.keras.layers.ReLU()(x)

    # Residual blocks
    for _ in range(9):
        x = residual_block(x)

    # Upsampler
    # First layer
    x = tf.keras.layers.Conv2DTranspose(128, 3, strides=2, padding='same', kernel_initializer=initializer, use_bias=False)(x)
    x = tfa.layers.InstanceNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    # Second layer
    x = tf.keras.layers.Conv2DTranspose(64, 3, strides=2, padding='same', kernel_initializer=initializer, use_bias=False)(x)
    x = tfa.layers.InstanceNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    # Third layer
    x = tf.keras.layers.Conv2D(output_channels, 7, strides=1, padding='same', kernel_initializer=initializer, use_bias=False)(x)
    x = tf.keras.layers.Activation('tanh')(x)

    return tf.keras.Model(inputs=inputs, outputs=x)


def discriminator():
    """PatchGAN discriminator, following architecture defined in report"""
    initializer = tf.random_normal_initializer(0., 0.02)
    inputs = tf.keras.layers.Input(shape=[256, 256, 3])
    x = inputs

    # First layer
    x = tf.keras.layers.Conv2D(64, 4, strides=2, padding='same', kernel_initializer=initializer, use_bias=False)(x)
    x = tf.keras.layers.LeakyReLU()(x)
    # Second layer
    x = tf.keras.layers.Conv2D(128, 4, strides=2, padding='same', kernel_initializer=initializer, use_bias=False)(x)
    x = tfa.layers.InstanceNormalization()(x)
    x = tf.keras.layers.LeakyReLU()(x)
    # Third layer
    x = tf.keras.layers.Conv2D(256, 4, strides=2, padding='same', kernel_initializer=initializer, use_bias=False)(x)
    x = tfa.layers.InstanceNormalization()(x)
    x = tf.keras.layers.LeakyReLU()(x)
    # Fourth layer
    x = tf.keras.layers.Conv2D(512, 4, strides=1, padding='same', kernel_initializer=initializer, use_bias=False)(x)
    x = tfa.layers.InstanceNormalization()(x)
    x = tf.keras.layers.LeakyReLU()(x)
    # Fifth layer
    x = tf.keras.layers.Conv2D(1, 4, strides=1, padding='same', kernel_initializer=initializer)(x)

    return tf.keras.Model(inputs=inputs, outputs=x)

