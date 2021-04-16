import models
import dataset
import tensorflow
from tensorflow.keras.layers import *
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from tensorflow.keras.optimizers import Adam

def get_data(feature_dir, label_dir):
    features, _ = dataset.load_face_pictures(feature_dir, color_mode="rgb")
    labels, _ = dataset.load_face_pictures(label_dir, color_mode="grayscale")
    features = features / 255
    labels = labels / 255

    return features, labels


if __name__ == '__main__':
    input_layer = Input((256, 256, 3))
    model = models.standard_unet(input_layer, 16)

    model.compile(Adam(), loss='binary_crossentropy', metrics=["accuracy"])


    epochs = 20000
    for epoch in range(epochs):
        features, labels = get_data(
            "train_data/medical/CelebA-HQ-img-256-256-masked",
            "train_data/medical/CelebA-HQ-img-256-256-labels")
        loss_acc = model.train_on_batch(features, labels)
        print("[Epoch %d/%d] [D loss_whole: %f, acc: %3d%%]", epoch, epochs, loss_acc[0], loss_acc[1])
        if epoch % 500 == 0:
            model.save("saved_models/" + str(epoch) + "segment_net", save_format="tf")
