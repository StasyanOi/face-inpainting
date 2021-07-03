import cv2.cv2 as cv2
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import Adam

import dataset
import models


def get_data(feature_dir, label_dir):
    features, labels = dataset.load_seg_data(feature_dir, label_dir, img_num=256)
    features = features / 255
    labels = labels / 255
    return features, labels


if __name__ == '__main__':
    input_layer = Input((256, 256, 3))
    model = models.segmentation_autoencoder(input_layer, 16)
    model.summary()
    model.compile(Adam(), loss=["binary_crossentropy"], metrics=["accuracy"])

    epochs = 20000
    for epoch in range(epochs):
        features, labels = get_data(
            "train_data/medical/CelebA-HQ-img-256-256-masked",
            "train_data/medical/CelebA-HQ-img-256-256-labels")
        loss_acc = model.train_on_batch(features, labels)
        print("[Epoch %d/%d] [D loss_whole: %f, acc_iou: %f]" % (epoch, epochs, loss_acc[0], loss_acc[1]))
        if epoch % 500 == 0:
            features, labels = get_data(
                "train_data/medical/CelebA-HQ-img-256-256-masked",
                "train_data/medical/CelebA-HQ-img-256-256-labels")
            predictions = model.predict(features)
            cv2.imwrite("gan_images/" + str(epoch) + "_segment.png", predictions[0] * 255)
            model.save("saved_models/" + str(epoch) + "segment_net", save_format="tf")
