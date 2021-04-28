from skimage.metrics import *
import cv2.cv2 as cv2

if __name__ == '__main__':
    real = cv2.imread("compare/metrics/real24300.png")
    generated = cv2.imread("compare/metrics/generated24300.png")
    print(structural_similarity(real, generated, multichannel=True))
    print(peak_signal_noise_ratio(real, generated))
