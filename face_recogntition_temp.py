import os

import cv2.cv2 as cv2
import face_recognition
import numpy as np


def sort_names(dir):
    ints = []
    for i in range(len(dir)):
        ints.append(int(dir[i].split(".")[0]))
    ints.sort()
    for i in range(len(dir)):
        dir[i] = str(ints[i]) + ".png"
    return dir


def equalize(image):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

    lab_planes = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(8, 8))

    lab_planes[0] = clahe.apply(lab_planes[0])

    lab = cv2.merge(lab_planes)

    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def face_recognize():
    known_image = cv2.imread("compare/me.png", cv2.IMREAD_COLOR)
    # known_image = equalize(known_image)
    known_encoding = face_recognition.face_encodings(known_image)[0]
    files = sort_names(os.listdir("inpaint_real/"))
    encodings = []
    imgs = []
    print("getting encodings")
    stop = len(files)
    for i in range(0, stop):
        unknown_image = cv2.imread("inpaint_real/" + files[i], cv2.IMREAD_COLOR)
        # unknown_image = equalize(unknown_image)
        imgs.append(unknown_image)
    for i in range(0, stop):
        face_encodings = face_recognition.face_encodings(imgs[i])
        if len(face_encodings) != 0:
            encodings.append(face_encodings[0])
    encodings = np.stack(encodings)
    results = face_recognition.compare_faces([known_encoding], encodings)
    print("calc probability")
    true = 0
    res_len = len(results)
    for i in range(res_len):
        if results[i]:
            true = true + 1

    true_stop = true / res_len

    if true_stop >= 0.5:
        print(1)
    elif 0.4 <= true_stop < 0.5:
        print(0.5)
    else:
        print(0)

if __name__ == '__main__':
    face_recognize()