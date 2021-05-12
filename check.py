import tensorflow as tf
import os
import cv2.cv2 as cv2


def detectAndDisplay(frame):
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(frame_gray)
    for (x, y, w, h) in faces:
        i = 3
        i1 = 20
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
    face_cascade_name = "haar/haarcascade_frontalface_alt2.xml"
    face_cascade = cv2.CascadeClassifier()
    if not face_cascade.load(cv2.samples.findFile(face_cascade_name)):
        print('--(!)Error loading face cascade')
        exit(0)

    cap = cv2.VideoCapture(0)
    i = 627
    while 1:
        ret, img = cap.read()
        img = detectAndDisplay(img)
        try:
            img = cv2.resize(img, (256, 256))
            cv2.imshow("img", img)
            cv2.imwrite("./me/" + str(i) + ".png", img)
            i = i + 1
            k = cv2.waitKey(30) & 0xff
            if k == 27:
                break
        except Exception as e:
            print(str(e))
