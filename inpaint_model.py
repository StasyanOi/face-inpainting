from __future__ import print_function, division

import datetime

import cv2.cv2 as cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.losses import mean_absolute_error
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Input, Dropout, Concatenate
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import UpSampling2D, Conv2D, GlobalAveragePooling2D, Reshape, Dense, Permute, multiply
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow_addons.layers import InstanceNormalization
import tensorflow.keras.backend as K
import PIL.Image as Image
from data_loader import DataLoader


class InpaintModel():

    def l1(self, y_true, y_pred):
        return mean_absolute_error(y_true, y_pred)

    def ssim_loss(self, y_true, y_pred):
        return 1 - tf.reduce_mean(tf.image.ssim(y_true, y_pred, 1.0))

    def generator_loss(self, y_true, y_pred):
        return self.l1(y_true, y_pred) + self.ssim_loss(y_true, y_pred)

    def __init__(self):
        self.img_rows = 256
        self.img_cols = 256
        self.channels = 3
        self.img_shape = (self.img_rows, self.img_cols, self.channels)

        # Configure data loader
        self.dataset_name = 'facades'
        self.data_loader = DataLoader(dataset_name=self.dataset_name,
                                      img_res=(self.img_rows, self.img_cols))

        # Calculate output shape of D (PatchGAN)
        patch = int(self.img_rows / 2 ** 4)
        self.disc_patch = (patch, patch, 1)

        # Number of filters in the first layer of G and D
        self.gf = 64
        self.df = 64

        optimizer = Adam(0.0002, 0.5)

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.summary()
        self.discriminator.compile(loss='binary_crossentropy',
                                   optimizer=optimizer,
                                   metrics=['accuracy'])

        self.discriminator_mask = self.build_discriminator()
        self.discriminator_mask.compile(loss='binary_crossentropy', loss_weights=[2],
                                        optimizer=optimizer,
                                        metrics=['accuracy'])
        # -------------------------
        # Construct Computational
        #   Graph of Generator
        # -------------------------

        # Build the generator
        self.generator = self.build_generator()
        self.generator.summary()

        # Input images and their conditioning images
        img_A = Input(shape=self.img_shape)
        img_B = Input(shape=self.img_shape)

        # By conditioning on B generate a fake version of A
        fake_A = self.generator(img_B)

        # For the combined model we will only train the generator
        self.discriminator.trainable = False
        self.discriminator_mask.trainable = False

        # Discriminators determines validity of translated images / condition pairs
        valid = self.discriminator([fake_A, img_B])
        valid_mask = self.discriminator_mask([fake_A, img_B])

        self.combined = Model(inputs=[img_A, img_B], outputs=[valid, valid_mask, fake_A])
        self.combined.compile(loss=['binary_crossentropy', 'binary_crossentropy', self.generator_loss],
                              loss_weights=[1, 2, 100],
                              optimizer=optimizer)

    def build_generator(self):
        """U-Net Generator"""

        def conv2d(layer_input, filters, f_size=4, bn=True):
            """Layers used during downsampling"""
            d = Conv2D(filters, kernel_size=f_size, strides=2, padding='same')(layer_input)
            d = LeakyReLU(alpha=0.2)(d)
            if bn:
                d = BatchNormalization(momentum=0.8)(d)
            return d

        def deconv2d(layer_input, skip_input, filters, f_size=4, dropout_rate=0):
            """Layers used during upsampling"""
            u = UpSampling2D(size=2)(layer_input)
            u = Conv2D(filters, kernel_size=f_size, strides=1, padding='same', activation='relu')(u)
            if dropout_rate:
                u = Dropout(dropout_rate)(u)
            u = BatchNormalization(momentum=0.8)(u)
            u = Concatenate()([u, skip_input])
            return u

        # Image input
        d0 = Input(shape=self.img_shape)

        # Downsampling
        d1 = conv2d(d0, self.gf, bn=False)
        d2 = conv2d(d1, self.gf * 2)
        d3 = conv2d(d2, self.gf * 4)
        d4 = conv2d(d3, self.gf * 8)
        d5 = conv2d(d4, self.gf * 8)
        d6 = conv2d(d5, self.gf * 16)
        d7 = conv2d(d6, self.gf * 16)

        # Upsampling
        u1 = deconv2d(d7, d6, self.gf * 8)
        u2 = deconv2d(u1, d5, self.gf * 8)
        u3 = deconv2d(u2, d4, self.gf * 8)
        u4 = deconv2d(u3, d3, self.gf * 4)
        u5 = deconv2d(u4, d2, self.gf * 2)
        u6 = deconv2d(u5, d1, self.gf)

        u7 = UpSampling2D(size=2)(u6)
        output_img = Conv2D(self.channels, kernel_size=4, strides=1, padding='same', activation='tanh')(u7)

        return Model(d0, output_img)

    def build_discriminator(self):

        def d_layer(layer_input, filters, f_size=4, bn=True):
            """Discriminator layer"""
            d = Conv2D(filters, kernel_size=f_size, strides=2, padding='same')(layer_input)
            d = LeakyReLU(alpha=0.2)(d)
            if bn:
                d = BatchNormalization(momentum=0.8)(d)
            return d

        img_A = Input(shape=self.img_shape)
        img_B = Input(shape=self.img_shape)

        # Concatenate image and conditioning image by channels to produce input
        combined_imgs = Concatenate(axis=-1)([img_A, img_B])

        d1 = d_layer(combined_imgs, self.df, bn=False)
        d2 = d_layer(d1, self.df * 2)
        d3 = d_layer(d2, self.df * 4)
        d4 = d_layer(d3, self.df * 8)

        validity = Conv2D(1, kernel_size=4, strides=1, padding='same')(d4)

        return Model([img_A, img_B], validity)

    def train(self, epochs, batch_size=64, sample_interval=100):

        start_time = datetime.datetime.now()

        # Adversarial loss ground truths
        valid = np.ones((batch_size,) + self.disc_patch)
        fake = np.zeros((batch_size,) + self.disc_patch)

        for epoch in range(epochs):
            (potential_output, input, mask) = self.data_loader.load_data(batch_size=batch_size)
            # ---------------------
            #  Train Discriminator
            # ---------------------

            # Condition on B and generate a translated version
            generated = self.generator.predict(input)

            # Train the discriminators (original images = real / generated = Fake)
            d_loss_real = self.discriminator.train_on_batch([potential_output, input], valid)
            d_loss_fake = self.discriminator.train_on_batch([generated, input], fake)

            d_loss_real_mask = self.discriminator_mask.train_on_batch([potential_output, input], valid)
            fake_A_mask = potential_output * (1 - mask) + generated * mask
            d_loss_fake_mask = self.discriminator_mask.train_on_batch([fake_A_mask, input], fake)

            d_loss_whole = 0.5 * np.add(d_loss_real, d_loss_fake)
            d_loss_mask = 0.5 * np.add(d_loss_real_mask, d_loss_fake_mask)

            # -----------------
            #  Train Generator
            # -----------------

            # Train the generators
            g_loss = self.combined.train_on_batch([potential_output, input], [valid, valid, potential_output])

            elapsed_time = datetime.datetime.now() - start_time
            # Plot the progress
            print("[Epoch %d/%d] [D loss_whole: %f, acc: %3d%%] [D loss_mask: %f, acc: %3d%%] [G loss: %f] time: %s" % (
            epoch, epochs,
            d_loss_whole[0],
            100 * d_loss_whole[1],
            d_loss_mask[0],
            100 * d_loss_mask[1],
            g_loss[0],
            elapsed_time))

            # If at save interval => save generated image samples
            if epoch % sample_interval == 0:
                self.sample_images(epoch)
                self.generator.save("saved_models/inpaint_net" + str(epoch), save_format="tf")

    def sample_images(self, epoch):
        imgs_A, imgs_B, _ = self.data_loader.load_data(batch_size=1)
        fake_A = self.generator.predict(imgs_B)

        cv2.imwrite("gan_images/real.png", ((0.5 * imgs_A[0] + 0.5) * 255).astype('uint8'))
        cv2.imwrite("gan_images/input.png", ((0.5 * imgs_B[0] + 0.5) * 255).astype('uint8'))
        cv2.imwrite("gan_images/generated.png", ((0.5 * fake_A[0] + 0.5) * 255).astype('uint8'))


if __name__ == '__main__':
    gan = InpaintModel()
    gan.train(epochs=80000, batch_size=64, sample_interval=100)
