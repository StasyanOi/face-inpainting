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


def load_face_pictures(dir, img_num=128, color_mode='grayscale'):
    # dataset = preprocessing.image_dataset_from_directory('dataset', color_mode='grayscale', image_size=(512, 512))
    dir_list = sort_names(os.listdir(dir))
    rands = np.random.randint(0, len(dir_list), img_num)
    files = [dir_list[rands[i]] for i in range(len(rands))]

    mode = -1
    if color_mode == 'grayscale':
        mode = 0
    else:
        mode = 1

    images = []
    for i in range(0, len(files)):
        input_arr_feature = cv2.imread(dir + "/" + files[i], mode)
        if mode == 0:
            input_arr_feature = np.resize(input_arr_feature, (256, 256, 1))
        elif mode == 1:
            input_arr_feature = np.resize(input_arr_feature, (256, 256, 3))
        elif mode == 1:
            alpha = (np.random.rand(1) * 2)
            input_arr_feature = cv2.convertScaleAbs(input_arr_feature, alpha=alpha[0], beta=0)

        images.append(input_arr_feature)

    batch_feature = np.stack(images)  # Convert single image to a batch.
    return batch_feature, files


def load_seg_data(feature_dir, label_dir, img_num=128):
    # dataset = preprocessing.image_dataset_from_directory('dataset', color_mode='grayscale', image_size=(512, 512))
    dir_list = sort_names(os.listdir(feature_dir))
    rands = np.random.randint(0, len(dir_list), img_num)
    files = [dir_list[rands[i]] for i in range(len(rands))]

    features, _ = load_face_pictures_list(feature_dir, files, color_mode="rgb")
    labels, _ = load_face_pictures_list(label_dir, files, color_mode="grayscale")

    return features, labels

def load_face_pictures_batch(dir, start, end, color_mode='grayscale'):
    # dataset = preprocessing.image_dataset_from_directory('dataset', color_mode='grayscale', image_size=(512, 512))
    dir_list = sort_names(os.listdir(dir))
    dir_list = dir_list[start:end]

    images = []
    for i in range(0, len(dir_list)):
        feature = tensorflow.keras.preprocessing.image.load_img(dir + "/" + dir_list[i],
                                                                color_mode=color_mode)
        input_arr_feature = tensorflow.keras.preprocessing.image.img_to_array(feature)
        images.append(input_arr_feature)

    batch_feature = np.array(images)  # Convert single image to a batch.
    return batch_feature, dir_list

def load_face_pictures_list(dir, lst, color_mode='grayscale'):
    # dataset = preprocessing.image_dataset_from_directory('dataset', color_mode='grayscale', image_size=(512, 512))

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
            alpha = (np.random.rand(1) * 2)
            input_arr_feature = cv2.convertScaleAbs(input_arr_feature, alpha=alpha[0], beta=0)
        images.append(input_arr_feature)

    batch_feature = np.array(images)  # Convert single image to a batch.
    return batch_feature, lst

def apply_mask(features, masks):
    for i in range(len(features)):
        feature = features[i]
        mask = masks[i]

        for j in range(feature.shape[2]):
            for k in range(feature.shape[0]):
                for m in range(feature.shape[1]):
                    if mask[k, m] == 255:
                        feature[k][m][j] = 255
        plt.imshow(features[0])
        plt.show()
    return np.copy(features)


def merge(features, masks):
    for i in range(len(features)):
        feature = features[i]
        mask = masks[i]

        for j in range(feature.shape[2]):
            for k in range(feature.shape[0]):
                for m in range(feature.shape[1]):
                    if mask[k, m] == 255:
                        feature[k][m][j] = 255

        merged_binary_face = "merged_binary_face"
        cv2.imwrite(merged_binary_face + "/" + str(i) + ".png", feature.astype('uint8'))
    return np.copy(features)


def merge_temp():
    feature = np.copy(np.asarray(Image.open("test_data/faces/face1/feature/feature_color_128_128.png")))
    label = np.copy(np.asarray(Image.open("test_data/faces/face1/label/label_face_256_256.png").resize((128, 128))))

    for j in range(feature.shape[2]):
        for k in range(feature.shape[0]):
            for m in range(feature.shape[1]):
                if label[k, m] == 255:
                    feature[k][m][j] = 255

    img = Image.fromarray(feature.astype('uint8'))
    img.save("test_data/faces/face1/merged/merged.png")


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
