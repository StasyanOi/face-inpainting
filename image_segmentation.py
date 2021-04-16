import models
import dataset
import tensorflow
import numpy as np
from tensorflow.keras.layers import *
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
import cv2.cv2 as cv2

def iou_coef(y_true, y_pred, smooth=1):
  intersection = K.sum(K.abs(y_true * y_pred), axis=[1,2,3])
  union = K.sum(y_true,[1,2,3])+K.sum(y_pred,[1,2,3])-intersection
  iou = K.mean((intersection + smooth) / (union + smooth), axis=0)
  return iou


def dice_coef(y_true, y_pred, smooth=1):
    """
    Dice = (2*|X & Y|)/ (|X|+ |Y|)
         =  2*sum(|A*B|)/(sum(A^2)+sum(B^2))
    ref: https://arxiv.org/pdf/1606.04797v1.pdf
    """
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    return (2. * intersection + smooth) / (K.sum(K.square(y_true),-1) + K.sum(K.square(y_pred),-1) + smooth)

def dice_coef_loss(y_true, y_pred):
    return 1 - dice_coef(y_true, y_pred)


def get_data(feature_dir, label_dir):
    features, labels = dataset.load_seg_data(feature_dir, label_dir, img_num=256)
    features = features / 255
    labels = labels / 255
    return features, labels


if __name__ == '__main__':
    input_layer = Input((256, 256, 3))
    model = models.standard_unet(input_layer, 16)

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
            cv2.imwrite("gan_images/" + str(epoch)+ "_segment.png", predictions[0] * 255)
            model.save("saved_models/" + str(epoch) + "segment_net", save_format="tf")
