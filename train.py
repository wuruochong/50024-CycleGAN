# CycleGAN training module
# Author: Ruochong Wu


import tensorflow as tf
import time

import model

from input import get_dataset
from loss import discriminator_loss, generator_loss, calc_cycle_loss, identity_loss

# Import constants
from constants import OUTPUT_CHANNELS, EPOCHS
AUTOTUNE = tf.data.AUTOTUNE


@tf.function
def train_step(real_x, real_y, generator_g, generator_f, discriminator_x, discriminator_y, generator_g_optimizer, generator_f_optimizer, discriminator_x_optimizer, discriminator_y_optimizer):
    """Train step"""
    # Create persistent tape
    with tf.GradientTape(persistent=True) as tape:
        # G: X -> Y
        # F: Y -> X
        # Generate fake images
        fake_y = generator_g(real_x, training=True)
        fake_x = generator_f(real_y, training=True)
        # Create cycled images
        cycled_x = generator_f(fake_y, training=True)
        cycled_y = generator_g(fake_x, training=True)
        # Generate identity images
        same_x = generator_f(real_x, training=True)
        same_y = generator_g(real_y, training=True)
        # Discriminator output
        disc_real_x = discriminator_x(real_x, training=True)
        disc_real_y = discriminator_y(real_y, training=True)
        disc_fake_x = discriminator_x(fake_x, training=True)
        disc_fake_y = discriminator_y(fake_y, training=True)

        # Calculate losses
        gen_g_loss = generator_loss(disc_fake_y)
        gen_f_loss = generator_loss(disc_fake_x)

        total_cycle_loss = calc_cycle_loss(real_x, cycled_x) + calc_cycle_loss(real_y, cycled_y)

        # Total generator loss = adversarial loss + cycle loss
        total_gen_g_loss = gen_g_loss + total_cycle_loss + identity_loss(real_y, same_y)
        total_gen_f_loss = gen_f_loss + total_cycle_loss + identity_loss(real_x, same_x)

        disc_x_loss = discriminator_loss(disc_real_x, disc_fake_x)
        disc_y_loss = discriminator_loss(disc_real_y, disc_fake_y)

    # Calculate gradients
    generator_g_gradients = tape.gradient(total_gen_g_loss, generator_g.trainable_variables)
    generator_f_gradients = tape.gradient(total_gen_f_loss, generator_f.trainable_variables)
    discriminator_x_gradients = tape.gradient(disc_x_loss, discriminator_x.trainable_variables)
    discriminator_y_gradients = tape.gradient(disc_y_loss, discriminator_y.trainable_variables)

    # Apply gradients
    generator_g_optimizer.apply_gradients(zip(generator_g_gradients, generator_g.trainable_variables))
    generator_f_optimizer.apply_gradients(zip(generator_f_gradients, generator_f.trainable_variables))
    discriminator_x_optimizer.apply_gradients(zip(discriminator_x_gradients, discriminator_x.trainable_variables))
    discriminator_y_optimizer.apply_gradients(zip(discriminator_y_gradients, discriminator_y.trainable_variables))


def train(dataset_name):
    """Train CycleGAN"""
    # Load dataset
    trainA, trainB, testA, testB = get_dataset(dataset_name)

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

    # Train loop
    for epoch in range(int(ckpt.completed_epochs), EPOCHS):
        start = time.time()
        print('Starting epoch {}'.format(epoch + 1))
        for image_x, image_y in tf.data.Dataset.zip((trainA, trainB)):
            train_step(image_x, image_y, generator_g, generator_f, discriminator_x, discriminator_y, generator_g_optimizer, generator_f_optimizer, discriminator_x_optimizer, discriminator_y_optimizer)
        ckpt.completed_epochs.assign_add(1)

        # Save checkpoint
        if (epoch + 1) % 20 == 0:
            ckpt_save_path = ckpt_manager.save()
            print('Saving checkpoint for epoch {} at {}'.format(epoch+1, ckpt_save_path))
        print('Time taken for epoch {} is {} sec\n'.format(epoch + 1, time.time()-start))

    # Return last generator
    return generator_g, generator_f


if __name__ == '__main__':

    # Train
    generator_g, generator_f = train('horse2zebra')
    # generator_g, generator_f = train('monet2photo')



