import cv2
import numpy as np
import os

def sort_names(dir):
    ints = []
    for i in range(len(dir)):
        ints.append(int(dir[i].split(".")[0]))
    ints.sort()
    for i in range(len(dir)):
        dir[i] = str(ints[i]) + ".png"
    return dir


if __name__ == '__main__':
    files = sort_names(os.listdir("inpaint_real"))
    img = []
    for i in range(0, len(files)):
        img.append(cv2.imread("inpaint_real/" + files[i]))

    height, width, layers = img[1].shape

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter('video.avi', fourcc, 1, (width, height))

    for j in range(len(img)):
        video.write(img[j])

    cv2.destroyAllWindows()
    video.release()
