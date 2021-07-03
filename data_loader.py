import os
import random

import dataset


class DataLoader():

    def get_randoms(self, batch=32):
        rands = []
        for i in range(0, batch):
            rands.append(random.randint(0, 9000))
        return rands

    def load_data(self, batch_size=64):
        labels = "train_data/medical/CelebA-HQ-img-256-256-labels"
        merged = "train_data/medical/CelebA-HQ-img-256-256-merged"
        masked = "train_data/medical/CelebA-HQ-img-256-256"
        indexes = self.get_randoms(batch=batch_size)
        dir_list = dataset.sort_names(os.listdir(merged))
        images = [dir_list[indexes[i]] for i in range(len(indexes))]

        input, _ = dataset.load_face_pictures_list_no_brightness(merged, images, color_mode="rgb")
        potential_output, _ = dataset.load_face_pictures_list_no_brightness(masked, images, color_mode="rgb")
        masks, _ = dataset.load_face_pictures_list_no_brightness(labels, images, color_mode="grayscale")

        input = input / 127.5 - 1.
        potential_output = potential_output / 127.5 - 1.
        masks = masks / 255

        return potential_output, input, masks
