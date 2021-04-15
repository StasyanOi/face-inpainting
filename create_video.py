import cv2
import numpy as np
import os

if __name__ == '__main__':
    files = os.listdir("inpaint_real")
    img = []
    for i in range(0, 99):
        img.append(cv2.imread("inpaint_real/" + str(i) + ".png"))

    height, width, layers = img[1].shape

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter('video.avi', fourcc, 1, (width, height))

    for j in range(len(img)):
        video.write(img[j])

    cv2.destroyAllWindows()
    video.release()
