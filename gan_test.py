from __future__ import print_function, division

from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Input, Dense, Reshape, Flatten, Dropout
from tensorflow.keras.layers import BatchNormalization, Activation, ZeroPadding2D
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import UpSampling2D, Conv2D
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.optimizers import Adam
from data_loader import DataLoader
import PIL.Image as Image
import matplotlib.pyplot as plt

import sys

import numpy as np

if __name__ == '__main__':
    dl = DataLoader()
    generator = load_model("saved_models/19900_inpaint", compile=False)
    # generator.summary()

    input = dl.load_img("train_data", "merge_centered_new_2.png")

    gen_imgs = generator.predict(input)

    for i in range(len(gen_imgs)):
        # Image.fromarray(((0.5 * potential_output[i] + 0.5) * 255).astype('uint8')).save("gan_images/real" + str(i) + ".png")
        Image.fromarray(((0.5 * input[i] + 0.5) * 255).astype('uint8')).save("gan_images/input" + str(i) + ".png")
        Image.fromarray(((0.5 * gen_imgs[i] + 0.5) * 255).astype('uint8')).save("gan_images/generated" + str(i) + ".png")

