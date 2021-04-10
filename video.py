import cv2.cv2 as cv2
from tensorflow.keras.models import load_model
import numpy as np
import PIL.Image as Image
import dataset
import shutil
import os

if __name__ == '__main__':
    shutil.rmtree("test_real")
    shutil.rmtree("results_real")
    shutil.rmtree("merged_real")
    shutil.rmtree("inpaint_real")
    os.mkdir("test_real")
    os.mkdir("results_real")
    os.mkdir("merged_real")
    os.mkdir("inpaint_real")
    model = load_model("saved_models/segment")
    inpaint = load_model("saved_models/13200_inpaint")
    print("loaded models")
    # model.summary()

    cap = cv2.VideoCapture(0)

    img_number = 200
    for i in range(img_number):
        ret, img = cap.read()
        img = img[:, 79:559, :]
        cv2.imshow('img.png', img)
        img = cv2.resize(img, dsize=(256, 256))
        cv2.imwrite("test_real/" + str(i) + '.png', img)
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break
    cap.release()
    cv2.destroyAllWindows()
    print("read input")

    masked_face_dir = "test_real"
    masked_faces, _ = dataset.load_face_pictures(masked_face_dir, img_num=img_number, color_mode='rgb')
    features = masked_faces / 255

    predictions = model.predict(features)
    predictions = np.round(predictions[:, :, :, 0]) * 255.0
    for i in range(len(predictions)):
        predictions[i] = cv2.erode(predictions[i], (1, 1))
        predictions[i] = cv2.dilate(predictions[i], (20, 20))
        cv2.imshow('img.jpg', (predictions[i]).astype("uint8"))
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break
        predicted_img = Image.fromarray((predictions[i]).astype("uint8"), "L")
        predicted_img.save("results_real/" + str(i) + ".png")
    print("done segmentation")
    dataset.merge_feature_mask("test_real", "results_real", "merged_real")

    faces_with_no_mask = "merged_real"
    features, _ = dataset.load_face_pictures(faces_with_no_mask, img_num=img_number, color_mode='rgb')
    features = features / 127.5 - 1
    inpaint_real = "inpaint_real"
    predictions = inpaint.predict(features)
    for i in range(img_number):
        predictions[i] = ((0.5 * predictions[i] + 0.5) * 255)
        cv2.imshow('img.jpg', (predictions[i]).astype("uint8"))
        k = cv2.waitKey(30) & 0xff
        predicted_img = Image.fromarray((predictions[i]).astype("uint8"))
        predicted_img.save(inpaint_real + "/" + str(i) + ".png")
    print("done inpainting")
