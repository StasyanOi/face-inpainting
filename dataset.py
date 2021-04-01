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
    return batch_feature


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


if __name__ == '__main__':
    merge_temp()
