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
    R, G, B = cv2.split(image)
    output1_R = cv2.equalizeHist(R)
    output1_G = cv2.equalizeHist(G)
    output1_B = cv2.equalizeHist(B)
    return cv2.merge((output1_R, output1_G, output1_B))


if __name__ == '__main__':

    known_image = face_recognition.load_image_file("compare/me.png")

    # known_image = equalize(known_image)

    known_encoding = face_recognition.face_encodings(known_image)[0]

    files = sort_names(os.listdir("compare/generated/"))
    img = []
    true = 0
    for i in range(0, len(files)):
        try:
            unknown_image = face_recognition.load_image_file("compare/generated/" + files[i])
            # unknown_image = equalize(unknown_image)
            unknown_encoding = face_recognition.face_encodings(unknown_image)[0]
            results = face_recognition.compare_faces([known_encoding], unknown_encoding)
            if (results[0] == True):
                true = true + 1
            print(i)
            print(true / len(files))
        except Exception as e:
            print(str(e))
