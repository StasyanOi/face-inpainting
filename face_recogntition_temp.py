import os
import random
from collections import Counter
import cv2.cv2 as cv2
import face_recognition
import numpy as np
import dataset

me_file = "me.png"

def equalize(image):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

    lab_planes = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(8, 8))

    lab_planes[0] = clahe.apply(lab_planes[0])

    lab = cv2.merge(lab_planes)

    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def get_randoms(batch=32):
    rands = []
    for i in range(0, batch):
        rands.append(random.randint(0, 9000))
    return rands


def get_random_faces():
    celeba = "train_data/medical/CelebA-HQ-img-256-256/"
    files = os.listdir(celeba)
    images = []
    rands = get_randoms(batch=1000)
    for i in range(len(rands)):
        images.append(cv2.imread(celeba + files[i], cv2.IMREAD_COLOR))
    etalon = "guf.png"
    etalon2 = me_file
    current_person_image = cv2.imread("compare/%s" % etalon, cv2.IMREAD_COLOR)
    current_person_image2 = cv2.imread("compare/%s" % etalon2, cv2.IMREAD_COLOR)
    images.append(current_person_image)
    images.append(current_person_image2)
    img_files = [files[i] for i in rands]
    img_files.append(etalon)
    img_files.append(etalon2)
    return np.array(images), img_files


def face_recognize():
    random_face_images, file_names = get_random_faces()
    encodings_faces = []
    for i in range(len(random_face_images)):
        face_encodings = face_recognition.face_encodings(random_face_images[i])
        if len(face_encodings) != 0:
            encodings_faces.append(face_encodings[0])
        else:
            file_names.remove(file_names[i])
    file_names = np.array(file_names)
    files = dataset.sort_names(os.listdir("inpaint_real/"))
    encodings = []
    imgs = []
    print("getting encodings")
    stop = len(files)
    for i in range(0, stop):
        unknown_image = cv2.imread("inpaint_real/" + files[i], cv2.IMREAD_COLOR)
        imgs.append(unknown_image)
    for i in range(0, stop):
        face_encodings = face_recognition.face_encodings(imgs[i])
        if len(face_encodings) != 0:
            encodings.append(face_encodings[0])
    encodings = np.stack(encodings)
    all_files = []
    for i in range(len(encodings)):
        results = np.array(face_recognition.compare_faces(encodings_faces, encodings[i]))
        matched_image_files = file_names[results]
        all_files.extend(matched_image_files)
    c = Counter(all_files)
    keys = list(c.keys())
    values = np.array(list(c.values())) / len(files)
    tuples = []

    for x, y in zip(keys, values):
        if me_file in x:
            if y > 0.5:
                print(1)
            elif 0.4 < y <= 0.5:
                print(0.5)
            elif y <= 0.4:
                print(0)
        tuples.append((y, x))

    if me_file not in keys:
        print(0)

    tuples.sort()
    print(tuples)


if __name__ == '__main__':
    face_recognize()