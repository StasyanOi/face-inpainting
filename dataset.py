import tensorflow.keras
import numpy as np
import matplotlib.pyplot as plt
import PIL.Image as Image
import os


def sort_names(dir):
    ints = []
    for i in range(len(dir)):
        ints.append(int(dir[i].split(".")[0]))
    ints.sort()
    for i in range(len(dir)):
        dir[i] = str(ints[i]) + ".png"
    return dir


def load_face_pictures(dir, img_num=1000, color_mode='grayscale'):
    # dataset = preprocessing.image_dataset_from_directory('dataset', color_mode='grayscale', image_size=(512, 512))
    dir_list = sort_names(os.listdir(dir))
    dir_list = dir_list[:img_num]

    images = []
    for i in range(0, len(dir_list)):
        feature = tensorflow.keras.preprocessing.image.load_img(dir + "/" + dir_list[i],
                                                                color_mode=color_mode)
        input_arr_feature = tensorflow.keras.preprocessing.image.img_to_array(feature)
        images.append(input_arr_feature)

    batch_feature = np.array(images)  # Convert single image to a batch.
    return batch_feature, dir_list


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
    for i in range(0, len(lst)):
        feature = tensorflow.keras.preprocessing.image.load_img(dir + "/" + lst[i],
                                                                color_mode=color_mode)
        input_arr_feature = tensorflow.keras.preprocessing.image.img_to_array(feature)
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

        img = Image.fromarray(feature.astype('uint8'))
        img.save("train_data/merged/" + str(i) + ".png")
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

    indexes = np.arange(0, 11000, 1000)

    for p in range(len(indexes) - 1):
        features, f_list = load_face_pictures_batch(masked, indexes[p], indexes[p + 1], color_mode='rgb')
        labels, l_list = load_face_pictures_batch(img_labels, indexes[p], indexes[p + 1])

        for i in range(len(features)):
            for j in range(features[i].shape[0]):
                for k in range(features[i].shape[1]):
                    if labels[i][j, k] == 255:
                        features[i][j, k, 0] = 255
                        features[i][j, k, 1] = 255
                        features[i][j, k, 2] = 255
            Image.fromarray(features[i].astype('uint8')).save(merged + f_list[i])


if __name__ == '__main__':
    merge_feature_mask()
