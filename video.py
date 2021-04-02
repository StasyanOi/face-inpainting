import cv2.cv2 as cv2
from tensorflow.keras.models import load_model
import numpy as np
import PIL.Image as Image

if __name__ == '__main__':
    model = load_model("unet_seg", compile=False)
    model.summary()

    cap = cv2.VideoCapture(0)
    while 1:
        ret, img = cap.read()
        img = img[:, 79:559, :]
        cv2.imshow('img.png', img)
        img = cv2.resize(img, dsize=(256, 256))
        cv2.imwrite('img1.png', img)
        img = img / 255
        predict = model.predict(np.array([img]))
        img = (np.round(predict[0, :, :, 0]) * 255.0).astype("uint8")

        cv2.imwrite('img2.png', img)
        # cv2.imshow('img.png', img)

        k = cv2.waitKey(30) & 0xff
        if k == 27:
            cv2.imwrite('img.png', img)
            break
    cap.release()
    cv2.destroyAllWindows()
