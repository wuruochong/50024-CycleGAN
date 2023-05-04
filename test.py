# CycleGAN testing module
# Author: Ruochong Wu


import tensorflow as tf
import os
import sys

import model

from input import get_dataset

# Import constants
from constants import OUTPUT_CHANNELS
AUTOTUNE = tf.data.AUTOTUNE


def generate_and_save(model, test_input, path, counter):
    """Generates image given current model, and saves it to provided directory"""
    prediction = model(test_input)
    in_path = path + str(counter) + '_in.png'
    out_path = path + str(counter) + '_out.png'
    tf.keras.preprocessing.image.save_img(in_path, test_input[0] * 0.5 + 0.5)
    tf.keras.preprocessing.image.save_img(out_path, prediction[0] * 0.5 + 0.5)
    return


def load_trained(dataset_name):
    """Loads trained model from checkpoint"""

    # Create generator and discriminator
    generator_g = model.generator(OUTPUT_CHANNELS)
    generator_f = model.generator(OUTPUT_CHANNELS)
    discriminator_x = model.discriminator()
    discriminator_y = model.discriminator()

    # Create optimizers
    generator_g_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
    generator_f_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
    discriminator_x_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
    discriminator_y_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

    # Create checkpoint manager, restore if checkpoint exists
    checkpoint_dir = './checkpoints/train/' + dataset_name
    ckpt = tf.train.Checkpoint(generator_g=generator_g,
                               generator_f=generator_f,
                               discriminator_x=discriminator_x,
                               discriminator_y=discriminator_y,
                               generator_g_optimizer=generator_g_optimizer,
                               generator_f_optimizer=generator_f_optimizer,
                               discriminator_x_optimizer=discriminator_x_optimizer,
                               discriminator_y_optimizer=discriminator_y_optimizer,
                               completed_epochs=tf.Variable(0))
    ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_dir, max_to_keep=5)

    if ckpt_manager.latest_checkpoint:
        ckpt.restore(ckpt_manager.latest_checkpoint)
        print('Latest checkpoint restored!!')
        completed_epochs = int(ckpt.completed_epochs)
        print('Trained for {} epochs'.format(completed_epochs))
        # Return loaded generator and epoch number
        return generator_g, generator_f, completed_epochs

    # If no checkpoint, exit
    else:
        print('No checkpoint found!!')
        sys.exit(0)


def test(dataset_name):
    """Tests trained model"""

    # Load dataset
    trainA, trainB, testA, testB = get_dataset(dataset_name)

    # Load trained model
    generator_g, generator_f, completed_epochs = load_trained(dataset_name)

    path = './outputs/' + dataset_name + '/epoch' + str(completed_epochs) + '/'
    pathA = path + 'A/'
    pathB = path + 'B/'
    if not os.path.exists(pathA):
        os.makedirs(pathA)
    if not os.path.exists(pathB):
        os.makedirs(pathB)

    # Test
    counter = 0
    for image_x, image_y in tf.data.Dataset.zip((testA, testB)):
        generate_and_save(generator_g, image_x, pathA, counter)
        generate_and_save(generator_f, image_y, pathB, counter)
        counter += 1


if __name__ == '__main__':
    test('horse2zebra')
    # test('monet2photo')

