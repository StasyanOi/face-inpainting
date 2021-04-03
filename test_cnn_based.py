import PIL.Image as Image
import numpy as np
import cv2.cv2 as cv2

from tensorflow.keras.models import load_model

import dataset

if __name__ == '__main__':

    # dir = "train_data/medical/CelebA-HQ-img-256-256-masked"
    dir = "test_real"
    features = dataset.load_face_pictures(dir, img_num=100, color_mode='rgb') / 255

    model = load_model("saved_models/unet_seg", compile=False)
    model.summary()
    predictions = model.predict(features)
    predictions = np.round(predictions[:, :, :, 0]) * 255.0
    for i in range(len(predictions)):
        cv2.imshow('img.jpg', (predictions[i]).astype("uint8"))
        k = cv2.waitKey(30) & 0xff
        predicted_img = Image.fromarray((predictions[i]).astype("uint8"), "L")
        # real_label = Image.fromarray((l_test[i] * 255).astype("uint8"), "RGB")
        predicted_img.save("results_real/" + str(i) + ".png")
        # real_label.save("results/real_" + str(i) + ".png")
