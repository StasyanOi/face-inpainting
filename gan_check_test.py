import cv2.cv2 as cv2
import numpy as np
from tensorflow.keras.metrics import BinaryAccuracy
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.metrics import *
import random
import dataset
import os


def photos():
    model = load_model("saved_models/500classifier_best")
    files = os.listdir("inpaint_real")
    images = []
    for i in range(len(files)):
        images.append(cv2.imread("inpaint_real/" + files[i], cv2.IMREAD_COLOR))
    me_pictures = np.stack(images)
    me_pictures = me_pictures / 255
    predicted = model.predict(me_pictures)
    predicted = np.round(predicted)
    ones = np.sum(predicted)
    print(ones / len(files))

def detectAndDisplay(frame):
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(frame_gray, scaleFactor=1.05)
    for (x, y, w, h) in faces:
        widen = 3
        up = 20
        x_ = int(x - w / widen)
        y_ = int(y - h / widen) - up
        x_w = int(x + w + w / widen)
        y_h = int(y + h + h / widen) - up
        frame = frame[y_:y_h, x_:x_w]
        try:
            frame = cv2.resize(frame, (256, 256))
        except Exception as e:
            print(str(e))
            frame = np.ones(shape=(256, 256, 3))
    return frame

if __name__ == '__main__':
    # photos()

    face_cascade_name = "haar/haarcascade_frontalface_alt2.xml"
    face_cascade = cv2.CascadeClassifier()
    if not face_cascade.load(cv2.samples.findFile(face_cascade_name)):
        print('--(!)Error loading face cascade')
        exit(0)

    model = load_model("saved_models/500classifier_best")
    cap = cv2.VideoCapture(0)
    while 1:
        ret, img = cap.read()
        img = detectAndDisplay(img)
        img = np.resize(img, (256, 256, 3))
        cv2.imshow("temp.png", img)
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break
        predicted = model.predict(np.array([img]) / 255)
        print(predicted[0])