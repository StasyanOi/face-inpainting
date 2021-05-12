import os
import dataset
import cv2

if __name__ == '__main__':
    files = dataset.sort_names(os.listdir("inpaint_real"))
    img = []
    for i in range(0, len(files)):
        img.append(cv2.imread("inpaint_real/" + files[i]))

    height, width, layers = img[1].shape

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter('videos/video.avi', fourcc, 1, (width, height))

    for j in range(len(img)):
        video.write(img[j])

    cv2.destroyAllWindows()
    video.release()
