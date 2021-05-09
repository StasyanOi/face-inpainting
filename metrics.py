from skimage.metrics import *
import cv2.cv2 as cv2
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
    files = sort_names(os.listdir("compare/metrics/custom/real/"))
    total_ssim = 0
    total_psnr = 0
    for i in range(len(files)):
        real = cv2.imread("compare/metrics/custom/real/" + files[i])
        generated = cv2.imread("compare/metrics/custom/generated/" + files[i])
        total_ssim = total_ssim + structural_similarity(real, generated, multichannel=True)
        total_psnr = total_psnr + peak_signal_noise_ratio(real, generated)

    print("ssim: " + str(total_ssim/len(files)))
    print("psnr: " + str(total_psnr/len(files)))
