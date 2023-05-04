# CycleGAN input pipeline
# Author: Ruochong Wu

import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt

# Define some constants
from constants import IMG_WIDTH, IMG_HEIGHT, BUFFER_SIZE, BATCH_SIZE
AUTOTUNE = tf.data.AUTOTUNE


def load_dataset(dataset_name):
    """Load dataset from tfds"""
    full_dataset_name = 'cycle_gan/' + dataset_name
    dataset, metadata = tfds.load(full_dataset_name, with_info=True, as_supervised=True)
    trainA, trainB = dataset['trainA'], dataset['trainB']
    testA, testB = dataset['testA'], dataset['testB']
    return trainA, trainB, testA, testB


def transform_train(image, label):
    """For transforming training images"""
    # Random jittering
    image = tf.image.resize(image, [286, 286], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    image = tf.image.random_crop(image, size=[IMG_HEIGHT, IMG_WIDTH, 3])
    # Random mirroring
    image = tf.image.random_flip_left_right(image)
    # Normalize to [-1, 1]
    image = tf.cast(image, tf.float32)
    image = (image / 127.5) - 1
    return image


def transform_test(image, label):
    """For transforming testing images"""
    # Resize to 256x256
    image = tf.image.resize(image, [IMG_HEIGHT, IMG_WIDTH], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    # Normalize to [-1, 1]
    image = tf.cast(image, tf.float32)
    image = (image / 127.5) - 1
    return image


def transform_dataset(trainA, trainB, testA, testB):
    """Transform datasets"""
    trainA = trainA.map(transform_train, num_parallel_calls=AUTOTUNE).cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
    trainB = trainB.map(transform_train, num_parallel_calls=AUTOTUNE).cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
    testA = testA.map(transform_test, num_parallel_calls=AUTOTUNE).cache().batch(BATCH_SIZE)
    testB = testB.map(transform_test, num_parallel_calls=AUTOTUNE).cache().batch(BATCH_SIZE)
    return trainA, trainB, testA, testB


def get_dataset(dataset_name):
    """Get dataset"""
    trainA, trainB, testA, testB = load_dataset(dataset_name)
    trainA, trainB, testA, testB = transform_dataset(trainA, trainB, testA, testB)
    return trainA, trainB, testA, testB


if __name__ == '__main__':
    trainA, trainB, testA, testB = get_dataset('apple2orange')
    for image in trainA.take(1):
        plt.imshow((image[0] + 1) / 2)
        plt.show()

