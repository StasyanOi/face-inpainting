import PIL.Image as Image
import numpy as np
import cv2.cv2 as cv2

from tensorflow.keras.models import load_model

import dataset

if __name__ == '__main__':

    # dir = "train_data/medical/CelebA-HQ-img-256-256-masked"
    dir = "test_real"
    features, _ = dataset.load_face_pictures(dir, img_num=100, color_mode='rgb')
    features = features / 255

    model = load_model("saved_models/segment")
    # model.summary()
    predictions = model.predict(features)
    predictions = np.round(predictions[:, :, :, 0]) * 255.0
    for i in range(len(predictions)):
        predictions[i] = cv2.morphologyEx(predictions[i], cv2.MORPH_OPEN, (5, 5))
        cv2.imshow('img.jpg', (predictions[i]).astype("uint8"))
        k = cv2.waitKey(30) & 0xff
        predicted_img = Image.fromarray((predictions[i]).astype("uint8"), "L")
        predicted_img.save("results_real/" + str(i) + ".png")
