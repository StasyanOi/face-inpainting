import face_recognition
import os
import cv2.cv2 as cv2

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


if __name__ == '__main__':

    known_image = cv2.imread("compare/me.png", cv2.IMREAD_COLOR)
    known_image = equalize(known_image)
    cv2.imwrite("cont_me.png", known_image)
    known_encoding = face_recognition.face_encodings(known_image)[0]

    files = sort_names(os.listdir("compare/generated/"))
    img = []
    true = 0
    for i in range(0, len(files)):
        try:
            unknown_image = cv2.imread("compare/generated/" + files[i], cv2.IMREAD_COLOR)
            unknown_image = equalize(unknown_image)
            cv2.imwrite("cont_me_un.png", unknown_image)
            unknown_encoding = face_recognition.face_encodings(unknown_image)[0]
            results = face_recognition.compare_faces([known_encoding], unknown_encoding)
            if (results[0] == True):
                true = true + 1
            print(i)
            print(true / len(files))
        except Exception as e:
            print(str(e))
