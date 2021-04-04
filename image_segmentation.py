import models
import dataset
import tensorflow
from tensorflow.keras.layers import *
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from tensorflow.keras.optimizers import Adam


class SaveModelCallback(tensorflow.keras.callbacks.Callback):
    def on_epoch_begin(self, epoch, logs=None):
        self.model.save("saved_models/unet_segment")


def get_data(feature_dir, label_dir):
    features, _ = dataset.load_face_pictures(feature_dir, color_mode="rgb")
    labels, _ = dataset.load_face_pictures(label_dir, color_mode="grayscale")
    features = features / 255
    labels = labels / 255
    features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.2)

    return features_train, features_test, labels_train, labels_test


if __name__ == '__main__':
    input_layer = Input((256, 256, 3))
    model = models.standard_unet(input_layer, 16)

    model.compile(Adam(), loss='binary_crossentropy', metrics=["accuracy"])

    tc = tensorflow.keras.callbacks.TensorBoard(log_dir="tensorboard")
    smc = SaveModelCallback()

    features_train, features_test, labels_train, labels_test = get_data(
        "train_data/medical/CelebA-HQ-img-256-256-masked",
        "train_data/medical/CelebA-HQ-img-256-256-labels")

    model.fit(features_train, labels_train, batch_size=64, callbacks=[tc, smc], validation_split=0.1, epochs=200)

    predicted_labels = model.predict(features_test)

    print(f1_score(labels_test, predicted_labels, average=None))
