from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from tensorflow.keras.utils import *
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.optimizers import *
from tensorflow.keras.datasets import *
from sklearn.model_selection import train_test_split
import PIL.Image as Image
import dataset
import models
import tensorflow as tf


def ssim_loss(y_true, y_pred):
    return 1 - tf.reduce_mean(tf.image.ssim(y_true, y_pred, 1.0))


if __name__ == '__main__':
    input_layer = Input((128, 128, 3))
    model = models.standard_unet(input_layer, 16)
    # model = load_model("inpaint_unet", compile=False)
    model.compile(Adam(), loss=ssim_loss, metrics=[ssim_loss, 'accuracy'])
    model.summary()

    features = dataset.load_face_pictures("train_data/balaclava/merged", img_num=731, color_mode='rgb')
    labels = dataset.load_face_pictures("train_data/balaclava/not_masked", img_num=731, color_mode='rgb')
    features = features / 255
    labels = labels / 255

    (f_train, f_test, l_train, l_test) = train_test_split(features, labels, test_size=0.2)

    tb = TensorBoard(log_dir="tensorboard")

    model.fit(f_train, l_train, validation_split=0.1, batch_size=32, epochs=50, callbacks=[tb])

    predictions = model.predict(f_test)

    model.save("inpaint_unet_new_mask")
