import os

import scipy
import dataset
import tensorflow
from glob import glob
import numpy as np
import matplotlib.pyplot as plt
import random


class DataLoader():
    def __init__(self, dataset_name="test", img_res=(128, 128)):
        self.dataset_name = dataset_name
        self.img_res = img_res

    def get_randoms(self, batch=32):
        rands = []
        for i in range(0, batch):
            rands.append(random.randint(0, 9000))
        return rands

    def load_data(self):
        labels = "train_data/medical/CelebA-HQ-img-256-256-labels"
        merged = "train_data/medical/CelebA-HQ-img-256-256-merged"
        masked = "train_data/medical/CelebA-HQ-img-256-256"
        indexes = self.get_randoms(batch=64)
        dir_list = dataset.sort_names(os.listdir(merged))
        images = [dir_list[indexes[i]] for i in range(len(indexes))]

        input, _ = dataset.load_face_pictures_list_no_brightness(merged, images, color_mode="rgb")
        potential_output, _ = dataset.load_face_pictures_list_no_brightness(masked, images, color_mode="rgb")
        masks, _ = dataset.load_face_pictures_list_no_brightness(labels, images, color_mode="grayscale")

        input = input / 127.5 - 1.
        potential_output = potential_output / 127.5 - 1.
        masks = masks / 255

        return potential_output, input, masks

    def load_img(self, dir, img_name):

        images = []
        feature = tensorflow.keras.preprocessing.image.load_img(dir + "/" + img_name, color_mode="rgb")
        input_arr_feature = tensorflow.keras.preprocessing.image.img_to_array(feature)
        images.append(input_arr_feature)

        batch_feature = np.array(images)  # Convert single image to a batch.

        batch_feature = batch_feature / 127.5 - 1
        return batch_feature

    # def load_batch(self, batch_size=1, is_testing=False):
    #     imgs_A = np.array(imgs_A) / 127.5 - 1.
    #     imgs_B = np.array(imgs_B) / 127.5 - 1.
    #
    #     return imgs_A, imgs_B

    def imread(self, path):
        return scipy.misc.imread(path, mode='RGB').astype(np.float)
