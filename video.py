import cv2.cv2 as cv2
import numpy
from tensorflow.keras.models import load_model
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
        i = 4
        i1 = 10
        x_ = int(x - w / i)
        y_ = int(y - h / i) - i1
        x_w = int(x + w + w / i)
        y_h = int(y + h + h / i) - i1
        frame = frame[y_:y_h, x_:x_w]
        try:
            frame = cv2.resize(frame, (256, 256))
        except Exception as e:
            print(str(e))
    return frame


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
    images = []
    img_number = 100
    for i in range(img_number):
        ret, img = cap.read()
        img = detectAndDisplay(img)
        try:
            cv2.imshow('img.png', img)
            images.append(img)
            k = cv2.waitKey(30) & 0xff
            if k == 27:
                break
        except Exception as e:
            print(str(e))
    cap.release()
    cv2.destroyAllWindows()
    print("read input")

    masked_faces = np.stack(images)

    features = masked_faces / 255

    predictions = model.predict(features)
    predictions = np.round(predictions[:, :, :, 0]) * 255.0
    for i in range(len(predictions)):
        predictions[i] = cv2.rotate(predictions[i], cv2.ROTATE_90_CLOCKWISE)
        predictions[i] = cv2.dilate(predictions[i], kernel=np.ones((10, 1)))
        predictions[i] = cv2.rotate(predictions[i], cv2.ROTATE_90_COUNTERCLOCKWISE)
        cv2.imshow('img.jpg', (predictions[i]).astype("uint8"))
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break

    print("done segmentation")
    features_merges = dataset.merge(masked_faces, predictions)

    features_merges = features_merges / 127.5 - 1
    inpaint_real = "inpaint_real"
    predictions = inpaint.predict(features_merges)
    for i in range(img_number):
        predictions[i] = ((0.5 * predictions[i] + 0.5) * 255)
        cv2.imshow('img.jpg', (predictions[i]).astype("uint8"))
        k = cv2.waitKey(30) & 0xff
        cv2.imwrite(inpaint_real + "/" + str(i) + ".png", (predictions[i]).astype("uint8"))
    print("done inpainting")
