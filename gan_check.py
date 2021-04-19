import numpy as np
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.metrics import *
import random
import dataset
import os

def discriminator():
    model = Sequential()

    kernel_initializer = "random_normal"
    kernel_regularizer = None
    dropout_rate = 0.4
    model.add(Conv2D(16, kernel_size=6, input_shape=(256, 256, 3), kernel_initializer=kernel_initializer,  kernel_regularizer=kernel_regularizer))
    model.add(LeakyReLU())
    model.add(Conv2D(8, kernel_size=4, kernel_initializer=kernel_initializer,  kernel_regularizer=kernel_regularizer))
    model.add(LeakyReLU())
    model.add(Conv2D(4, kernel_size=4, kernel_initializer=kernel_initializer,  kernel_regularizer=kernel_regularizer))
    model.add(LeakyReLU())
    model.add(Conv2D(1, kernel_size=4, kernel_initializer=kernel_initializer,  kernel_regularizer=kernel_regularizer))
    model.add(Flatten())
    model.add(LeakyReLU())
    model.add(Dropout(dropout_rate))
    model.add(Dense(1, kernel_initializer=kernel_initializer))
    model.add(LeakyReLU())

    model.compile(Adam(), "binary_crossentropy", metrics=['accuracy'])
    model.summary()

    return model

def get_randoms(batch=32, lower_bound=0, upper_bound=9999, ):
    rands = []
    for i in range(0, batch):
        rands.append(random.randint(lower_bound, upper_bound))
    return rands

def load_data(everyone_dir, me_dir, batch):
    indexes = get_randoms(batch=batch, lower_bound=0, upper_bound=600)
    me = dataset.sort_names(os.listdir(me_dir))
    everyone = dataset.sort_names(os.listdir(everyone_dir))
    images_me = [me[indexes[i]] for i in range(len(indexes))]
    images_everyone = [everyone[indexes[i]] for i in range(len(indexes))]

    me_pictures, _ = dataset.load_face_pictures_list(me_dir, images_me, color_mode="rgb")
    everyone_pictures, _ = dataset.load_face_pictures_list(everyone_dir, images_everyone, color_mode="rgb")

    return me_pictures, everyone_pictures


if __name__ == '__main__':
    model = discriminator()

    batch = 64

    me = np.ones(shape=(batch,))
    not_me = np.zeros(shape=(batch,))

    epochs = 200

    for epoch in range(epochs):
        me_pictures, everyone_pictures = load_data(everyone_dir="./train_data/medical/CelebA-HQ-img-256-256",
                                                   me_dir="./me", batch=batch)

        me_pictures = me_pictures / 255
        everyone_pictures = everyone_pictures / 255

        features = np.concatenate((me_pictures, everyone_pictures))
        labels = np.concatenate((me, not_me))

        la = model.train_on_batch(features, labels)

        print("[Epoch %d/%d] [loss: %f, acc: %f]" % (
            epoch, epochs,
            la[0], la[1]))

        if epoch % 10 == 0:
            model.save("./saved_models/" + str(epoch) + "classifier", save_format="tf")



