import tensorflow.keras
import numpy as np
import matplotlib.pyplot as plt
import PIL.Image as Image
import os
import cv2.cv2 as cv2


def sort_names(dir):
    ints = []
    for i in range(len(dir)):
        ints.append(int(dir[i].split(".")[0]))
    ints.sort()
    for i in range(len(dir)):
        dir[i] = str(ints[i]) + ".png"
    return dir


def load_seg_data(feature_dir, label_dir, img_num=128):
    dir_list = sort_names(os.listdir(feature_dir))
    rands = np.random.randint(0, len(dir_list), img_num)
    files = [dir_list[rands[i]] for i in range(len(rands))]

    features, _ = load_face_pictures_list(feature_dir, files, color_mode="rgb")
    labels, _ = load_face_pictures_list(label_dir, files, color_mode="grayscale")

    return features, labels


def load_face_pictures_list(dir, lst, color_mode='grayscale'):
    images = []
    mode = -1
    if color_mode == 'grayscale':
        mode = 0
    else:
        mode = 1

    for i in range(0, len(lst)):
        input_arr_feature = cv2.imread(dir + "/" + lst[i], mode)
        if mode == 0:
            input_arr_feature = np.resize(input_arr_feature, (256, 256, 1))
        elif mode == 1:
            input_arr_feature = np.resize(input_arr_feature, (256, 256, 3))
            alpha = (np.random.rand(1) * 2)
            input_arr_feature = cv2.convertScaleAbs(input_arr_feature, alpha=alpha[0], beta=0)
        images.append(input_arr_feature)

    batch_feature = np.array(images)  # Convert single image to a batch.
    return batch_feature, lst


def load_face_pictures_list_no_brightness(dir, lst, color_mode='grayscale'):
    images = []

    mode = -1
    if color_mode == 'grayscale':
        mode = 0
    else:
        mode = 1

    for i in range(0, len(lst)):
        input_arr_feature = cv2.imread(dir + "/" + lst[i], mode)
        if mode == 0:
            input_arr_feature = np.resize(input_arr_feature, (256, 256, 1))
        elif mode == 1:
            input_arr_feature = np.resize(input_arr_feature, (256, 256, 3))
        images.append(input_arr_feature)

    batch_feature = np.array(images)  # Convert single image to a batch.
    return batch_feature, lst


def merge(features, masks):
    for i in range(len(features)):
        feature = features[i]
        mask = masks[i].astype('uint8')

        inverted = (np.invert(mask) / 255).astype('uint8')

        feature[:, :, 0] = feature[:, :, 0] * inverted + mask
        feature[:, :, 1] = feature[:, :, 1] * inverted + mask
        feature[:, :, 2] = feature[:, :, 2] * inverted + mask

        merged_binary_face = "merged_binary_face"
        cv2.imwrite(merged_binary_face + "/" + str(i) + ".png", feature.astype('uint8'))
    return np.copy(features)


def merge_feature_mask(masked_people="./train_data/medical/CelebA-HQ-img-256-256-masked",
                       binary_labels="./train_data/medical/CelebA-HQ-img-256-256-labels",
                       merged_dir="./train_data/medical/CelebA-HQ-img-256-256-merged"):
    masked = masked_people + "/"
    img_labels = binary_labels + "/"
    merged = merged_dir + "/"

    dir_list = sort_names(os.listdir(masked))

    indexes = np.arange(0, 15000, 1000)

    for p in range(len(indexes) - 1):
        files = [dir_list[i] for i in range(indexes[p + 1])]
        features, f_list = load_face_pictures_list(masked, files, color_mode='rgb')
        labels, l_list = load_face_pictures_list(img_labels, files)

        for i in range(len(features)):
            for j in range(features[i].shape[0]):
                for k in range(features[i].shape[1]):
                    if labels[i][j, k] == 255:
                        features[i][j, k, 0] = 255
                        features[i][j, k, 1] = 255
                        features[i][j, k, 2] = 255
            cv2.imwrite(merged + f_list[i], features[i].astype('uint8'))


def merge_features(masked_people="./train_data/medical/CelebA-HQ-img-256-256-masked",
                   binary_labels="./train_data/medical/CelebA-HQ-img-256-256-labels",
                   merged_dir="./train_data/medical/CelebA-HQ-img-256-256-merged"):
    masked = masked_people + "/"
    img_labels = binary_labels + "/"
    merged = merged_dir + "/"

    dir_list = sort_names(os.listdir(masked))

    features, f_list = load_face_pictures_list(masked, dir_list, color_mode='rgb')
    labels, l_list = load_face_pictures_list(img_labels, dir_list)

    for i in range(len(features)):
        for j in range(features[i].shape[0]):
            for k in range(features[i].shape[1]):
                if labels[i][j, k] == 255:
                    features[i][j, k, 0] = 255
                    features[i][j, k, 1] = 255
                    features[i][j, k, 2] = 255
        cv2.imwrite(merged + f_list[i], features[i].astype('uint8'))


if __name__ == '__main__':
    merge_feature_mask()
