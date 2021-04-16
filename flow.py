import cv2.cv2 as cv2
import numpy
from tensorflow.keras.models import load_model
import tensorflow
import numpy as np
import PIL.Image as Image
import dataset
import shutil
import os


def detectAndDisplay(frame):
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_gray = cv2.equalizeHist(frame_gray)
    faces = face_cascade.detectMultiScale(frame_gray)
    for (x, y, w, h) in faces:
        x_ = int(x - w / 3)
        y_ = int(y - h / 3) - 20
        x_w = int(x + w + w / 3)
        y_h = int(y + h + h / 3) - 20
        frame = frame[y_:y_h, x_:x_w]
        try:
            frame = cv2.resize(frame, (256, 256))
        except Exception as e:
            print(str(e))
            frame = np.ones(shape=(256, 256, 3))
    return frame


def rearrange_channels(img_init):
    r = img_init[:, :, 0]
    g = img_init[:, :, 1]
    b = img_init[:, :, 2]

    img_init[:, :, 0] = b
    img_init[:, :, 1] = g
    img_init[:, :, 2] = r

    return img_init


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
    inpaint = load_model("saved_models/19900_inpaint")
    print("loaded models")
    # model.summary()

    face_cascade_name = "haar/haarcascade_frontalface_alt.xml"
    face_cascade = cv2.CascadeClassifier()
    if not face_cascade.load(cv2.samples.findFile(face_cascade_name)):
        print('--(!)Error loading face cascade')
        exit(0)

    cap = cv2.VideoCapture(0)

    while 1:
        ret, img = cap.read()
        img_init = detectAndDisplay(img)
        # img_init = rearrange_channels(img_init)
        cv2.imwrite("temp.png", img_init)
        img = cv2.imread("temp.png")
        img_ = np.array([img]) / 255
        predict = model.predict(img_)

        prediction = (np.round(predict[:, :, :, 0]) * 255.0).astype('uint8')

        for j in range(img.shape[0]):
            for k in range(img.shape[1]):
                if prediction[0][j, k] == 255:
                    img[j, k, 0] = 255
                    img[j, k, 1] = 255
                    img[j, k, 2] = 255

        img = (img / 127.5) - 1

        prediction = inpaint.predict(np.array([img]))
        prediction = ((0.5 * prediction + 0.5) * 255)

        try:
            cv2.imshow('image', prediction[0].astype('uint8'))
            k = cv2.waitKey(30) & 0xff
            if k == 27:
                break
        except Exception as e:
            print(str(e))

    cap.release()
