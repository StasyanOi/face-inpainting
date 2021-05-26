import math

import cv2
import numpy as np
import dlib


def shape_to_np(shape, dtype="int"):
    coords = np.zeros((68, 2), dtype=dtype)
    for i in range(0, 68):
        coords[i] = (shape.part(i).x, shape.part(i).y)
    return coords


face_cascade = cv2.CascadeClassifier("haar/haarcascade_frontalface_default.xml")
predictor = dlib.shape_predictor("haar/shape_predictor_68_face_landmarks.dat")


def getFace(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    # loading the classifiers with respected files
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.10, minNeighbors=5)
    face = None
    # looping through each detected faces and drawing rectangle around the face and circles around the feature points
    if len(faces) > 0:
        for x, y, w, h in faces:
            # cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 3)
            # creating the rectangle object from the outputs of haar cascade calssifier
            drect = dlib.rectangle(int(x), int(y), int(x + w), int(y + h))
            landmarks = predictor(gray, drect)
            points = shape_to_np(landmarks)

            left = points[36:42]
            right = points[42:48]

            x1 = int(np.mean(left[:, 0]))
            y1 = int(np.mean(left[:, 1]))

            x2 = int(np.mean(right[:, 0]))
            y2 = int(np.mean(right[:, 1]))

            dY = y2 - y1
            dX = x2 - x1
            angle = np.degrees(np.arctan2(dY, dX))

            eyesCenter = ((x1 + x2) // 2, (y1 + y2) // 2)

            rot_mat = cv2.getRotationMatrix2D(eyesCenter, angle, 1.0)
            result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)

            gray_new = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
            faces_new = face_cascade.detectMultiScale(gray_new, scaleFactor=1.10, minNeighbors=5)
            for x_n, y_n, w_n, h_n in faces_new:
                drect = dlib.rectangle(int(x), int(y), int(x + w), int(y + h))
                landmarks = predictor(gray_new, drect)
                points = shape_to_np(landmarks)

                left = points[36:42]
                right = points[42:48]

                x1 = int(np.mean(left[:, 0]))
                y1 = int(np.mean(left[:, 1]))

                x2 = int(np.mean(right[:, 0]))
                y2 = int(np.mean(right[:, 1]))

                dY = y2 - y1
                dX = x2 - x1

                between_eyes = math.sqrt((dY ** 2 + dX ** 2))

                width = between_eyes * 4

                up = width * 0.473
                down = width * 0.5273
                side = width * 0.3789

                left_side = int(x1 - side)
                right_side = int(x2 + side)
                up_side = int(y1 - up)
                down_side = int(y1 + down)

                result = result[up_side:down_side, left_side:right_side]
                try:
                    face = cv2.resize(result, (256, 256))
                except Exception as e:
                    print(str(e))
    return face


eye_cascade = cv2.CascadeClassifier("haar/haarcascade_eye.xml")

dialate = 15
widen = 3
up = 20


def detectAndDisplay(frame):
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_gray = cv2.equalizeHist(frame_gray)
    faces = face_cascade.detectMultiScale(frame_gray)
    eyes = eye_cascade.detectMultiScale(frame_gray)
    for (x, y, w, h) in faces:
        i = widen
        i1 = up
        x_ = int(x - w / i)
        y_ = int(y - h / i) - i1
        x_w = int(x + w + w / i)
        y_h = int(y + h + h / i) - i1
        frame = frame[y_:y_h, x_:x_w]
        index = 0
        for (eye_x, eye_y, eye_w, eye_h) in eyes:
            if index == 0:
                eye_1 = (eye_x, eye_y, eye_w, eye_h)
            elif index == 1:
                eye_2 = (eye_x, eye_y, eye_w, eye_h)

            cv2.rectangle(frame, (eye_x, eye_y), (eye_x + eye_w, eye_y + eye_h), 2)
            index = index + 1
        try:
            frame = cv2.resize(frame, (256, 256))
        except Exception as e:
            print(str(e))
    return frame


if __name__ == '__main__':
    cap = cv2.VideoCapture(0)
    while 1:
        ret, image = cap.read()
        face = getFace(image)

        cv2.imshow('img', face)
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break
