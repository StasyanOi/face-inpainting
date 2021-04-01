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

if __name__ == '__main__':

    dir = "train_data/balaclava/merged"
    features = dataset.load_face_pictures(dir, img_num=100, color_mode='rgb') / 255

    model = load_model("saved_models/inpaint_unet_new_mask", compile=False)
    model.summary()
    predictions = model.predict(features)

    for i in range(len(predictions)):
        predicted_img = Image.fromarray((predictions[i] * 255).astype("uint8"), "RGB")
        # real_label = Image.fromarray((l_test[i] * 255).astype("uint8"), "RGB")
        predicted_img.save("results/" + str(i) + ".png")
        # real_label.save("results/real_" + str(i) + ".png")
