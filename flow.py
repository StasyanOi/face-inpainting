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
    model = load_model("saved_models/2500segment_net")
    inpaint = load_model("saved_models/19200inpaint_net")
    print("loaded models")
    # model.summary()

    face_cascade_name = "haar/haarcascade_frontalface_alt2.xml"
    face_cascade = cv2.CascadeClassifier()
    if not face_cascade.load(cv2.samples.findFile(face_cascade_name)):
        print('--(!)Error loading face cascade')
        exit(0)

    cap = cv2.VideoCapture(0)

    while 1:
        ret, img = cap.read()
        img = detectAndDisplay(img)
        cv2.imwrite("temp.png", img)
        img_ = np.array([img]) / 255
        predict = None
        try:
            predict = model.predict(img_)
            prediction = (np.round(predict[0, :, :, :]) * 255.0).astype('uint8')
            cv2.imwrite("temp_1.png", prediction.reshape((256,256,1)))
            prediction = cv2.rotate(prediction, cv2.ROTATE_90_CLOCKWISE)
            prediction = cv2.dilate(prediction, kernel=np.ones((20,1)))
            prediction = cv2.rotate(prediction, cv2.ROTATE_90_COUNTERCLOCKWISE)
            cv2.imwrite("temp_2.png", prediction.reshape((256,256,1)))
            for j in range(img.shape[0]):
                for k in range(img.shape[1]):
                    if prediction[j, k] == 255:
                        img[j, k, 0] = 255
                        img[j, k, 1] = 255
                        img[j, k, 2] = 255

            img = (img / 127.5) - 1

            prediction = inpaint.predict(np.array([img]))
            prediction = ((0.5 * prediction + 0.5) * 255)

            try:
                pred = prediction[0].astype('uint8')
                cv2.imshow('image', pred)
                cv2.imwrite("temp_gen.png", pred)
                k = cv2.waitKey(30) & 0xff
                if k == 27:
                    break
            except Exception as e:
                print(str(e))
        except Exception as e:
            print(e)

    cap.release()
