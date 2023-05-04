# CycleGAN Loss Function Module
# Author: Ruochong Wu

import tensorflow as tf

# Import constants
from constants import LAMBDA


def discriminator_loss(real, generated):
    """Calculate discriminator loss"""
    real_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.NONE)(tf.ones_like(real), real)
    generated_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.NONE)(tf.zeros_like(generated), generated)
    return (real_loss + generated_loss) * 0.5


def generator_loss(generated):
    """Calculate generator loss"""
    return tf.keras.losses.BinaryCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.NONE)(tf.ones_like(generated), generated)


def calc_cycle_loss(real_image, cycled_image):
    """Calculate cycle loss"""
    loss1 = tf.reduce_mean(tf.abs(real_image - cycled_image))
    return LAMBDA * loss1


def identity_loss(real_image, same_image):
    """Calculate identity loss"""
    loss = tf.reduce_mean(tf.abs(real_image - same_image))
    return LAMBDA * 0.5 * loss

