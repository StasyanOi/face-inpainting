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

if __name__ == '__main__':
    cap = cv2.VideoCapture(0)
    while 1:
        ret, image = cap.read()
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # loading the classifiers with respected files
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.10, minNeighbors=5)
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

                between_eyes = math.sqrt((dY ** 2 + dX ** 2))




                eyesCenter = ((x1 + x2) // 2, (y1 + y2) // 2)

                rot_mat = cv2.getRotationMatrix2D(eyesCenter, angle, 1.0)
                result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)

                gray_new = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
                faces_new = face_cascade.detectMultiScale(gray_new, scaleFactor=1.10, minNeighbors=5)
                if len(faces) > 0:
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
                        print(between_eyes)

                        width = between_eyes * 4

                        up = width * 0.473
                        down = width * 0.5273
                        side = width * 0.3789

                        left_side = int(x1 - side)
                        right_side = int(x2 + side)
                        up_side = int(y1 - up)
                        down_side = int(y1 + down)

                        result = result[up_side:down_side, left_side:right_side]


            cv2.imshow('img', result)
            cv2.imwrite('img.png', result)
            k = cv2.waitKey(30) & 0xff
            if k == 27:
                break

cap.release()
cv2.destroyAllWindows()
