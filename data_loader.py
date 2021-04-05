import os

import scipy
import dataset
from glob import glob
import numpy as np
import matplotlib.pyplot as plt
import random


class DataLoader():
    def __init__(self, dataset_name, img_res=(128, 128)):
        self.dataset_name = dataset_name
        self.img_res = img_res

    def get_randoms(self, batch=32):
        rands = []
        for i in range(0, batch):
            rands.append(random.randint(0, 10000))
        return rands

    def load_data(self):
        merged = "medical/CelebA-HQ-img-256-256-merged"
        masked = "medical/CelebA-HQ-img-256-256"
        indexes = self.get_randoms(batch=128)
        dir_list = dataset.sort_names(os.listdir(merged))
        images = [dir_list[indexes[i]] for i in range(len(indexes))]

        input = dataset.load_face_pictures_list(merged, images, color_mode="rgb")
        potential_output = dataset.load_face_pictures_list(masked, images, color_mode="rgb")

        input = input / 127.5 - 1.
        potential_output = potential_output / 127.5 - 1.

        return potential_output, input

    # def load_batch(self, batch_size=1, is_testing=False):
    #     imgs_A = np.array(imgs_A) / 127.5 - 1.
    #     imgs_B = np.array(imgs_B) / 127.5 - 1.
    #
    #     return imgs_A, imgs_B

    def imread(self, path):
        return scipy.misc.imread(path, mode='RGB').astype(np.float)
