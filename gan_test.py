from __future__ import print_function, division

import os

import cv2.cv2 as cv2
import numpy as np
from tensorflow.keras.models import load_model

import dataset
from data_loader import DataLoader

if __name__ == '__main__':
    dl = DataLoader()
    generator = load_model("saved_models/25100inpaint_net")
    model = load_model("saved_models/1000segment_net")
    masked_ = "train_data/medical/CelebA-HQ-img-256-256-masked/"
    real_ = "train_data/medical/CelebA-HQ-img-256-256/"
    files = dataset.sort_names(os.listdir(masked_))
    images = []
    files_names = []
    for i in range(500):
        i_ = files[15000 + i]
        files_names.append(i_)
        imread = cv2.imread(masked_ + i_)
        images.append(imread)

    features_init = np.stack(images)

    features_for_mask = features_init / 255

    predictions_mask = model.predict(features_for_mask)
    predictions_mask = np.round(predictions_mask[:, :, :, 0]) * 255.0
    predictions = dataset.merge(features_init, predictions_mask)
    features = predictions / 127.5 - 1

    predictions = generator.predict(features)

    for i in range(len(predictions)):
        cv2.imwrite("compare/metrics/custom/generated/" + str(i) + ".png",
                    ((0.5 * predictions[i] + 0.5) * 255).astype('uint8'))
        cv2.imwrite("compare/metrics/custom/real/" + str(i) + ".png",
                    cv2.imread("train_data/medical/CelebA-HQ-img-256-256/"
                               + files_names[i]))
