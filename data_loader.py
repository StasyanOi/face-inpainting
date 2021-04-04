import scipy
import dataset
from glob import glob
import numpy as np
import matplotlib.pyplot as plt


class DataLoader():
    def __init__(self, dataset_name, img_res=(128, 128)):
        self.dataset_name = dataset_name
        self.img_res = img_res

    def load_data(self):
        merged = "medical/CelebA-HQ-img-256-256-merged"
        masked = "medical/CelebA-HQ-img-256-256"

        input, _ = dataset.load_face_pictures_batch(merged, 0, 10, color_mode="rgb")
        potential_output, _ = dataset.load_face_pictures_batch(masked, 0, 10, color_mode="rgb")

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
