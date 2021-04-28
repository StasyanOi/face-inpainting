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
import cv2.cv2 as cv2
import matplotlib.pyplot as plt

import sys

import numpy as np

if __name__ == '__main__':
    dl = DataLoader()
    generator = load_model("saved_models/25100inpaint_net", compile=False)
    # generator.summary()

    images = []
    for i in range(100):
        images.append(cv2.imread("train_data/medical/CelebA-HQ-img-256-256-merged/" + str(i) + ".png"))

    features = np.stack(images) / 127.5 - 1

    predictions = generator.predict(features)

    for i in range(len(predictions)):
        cv2.imwrite("compare/metrics/custom/generated/" + str(i) + ".png", ((0.5 * predictions[i] + 0.5) * 255).astype('uint8'))
        cv2.imwrite("compare/metrics/custom/real/" + str(i) + ".png", cv2.imread("train_data/medical/CelebA-HQ-img-256-256/" + str(i) + ".png"))

