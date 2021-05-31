import os
import shutil

import cv2.cv2 as cv2
import numpy as np
from tensorflow.keras.models import load_model

import dataset
import detection
import face_recogntition_system


class InpaintSystem:

    def __init__(self, segment_model=None, inpaint_model=None):
        print("Setting up environment")

        shutil.rmtree("video_capture")
        shutil.rmtree("face_segmentation")
        shutil.rmtree("binary_segmentation")
        shutil.rmtree("merged_binary_face")
        shutil.rmtree("inpaint_real")
        os.mkdir("video_capture")
        os.mkdir("face_segmentation")
        os.mkdir("binary_segmentation")
        os.mkdir("merged_binary_face")
        os.mkdir("inpaint_real")

        print("Loading Models")
        self.segment_model = load_model("saved_models/1000segment_net")
        self.inpaint_model = load_model("saved_models/25100inpaint_net")
        print("Models loaded")

    def face_segment(self, video_frames):
        print("Get faces")
        face_segmentation = "face_segmentation"
        faces = []
        for i in range(len(video_frames)):
            img = video_frames[i]
            img = detection.getFace(img)
            if img is not None:
                cv2.imshow('face', img)
                cv2.imwrite(face_segmentation + "/" + str(i) + ".png", img.astype("uint8"))
                faces.append(img)
                k = cv2.waitKey(30) & 0xff
                if k == 27:
                    break
        return np.stack(faces)

    def binary_segmentation(self, face_segments):
        print("Binary segment")
        face_segments_normalized = face_segments / 255
        binary_segments = self.segment_model.predict(face_segments_normalized)
        binary_segments = np.round(binary_segments[:, :, :, 0]) * 255.0
        binary_segmentation_dir = "binary_segmentation"
        for i in range(len(binary_segments)):
            binary_segments[i] = cv2.rotate(binary_segments[i], cv2.ROTATE_90_CLOCKWISE)
            binary_segments[i] = cv2.dilate(binary_segments[i], kernel=np.ones((detection.dialate, 1)))
            binary_segments[i] = cv2.rotate(binary_segments[i], cv2.ROTATE_90_COUNTERCLOCKWISE)
            cv2.imwrite(binary_segmentation_dir + "/" + str(i) + ".png", binary_segments[i].astype("uint8"))
            cv2.imshow('img.jpg', (binary_segments[i]).astype("uint8"))
            k = cv2.waitKey(30) & 0xff
            if k == 27:
                break
        return binary_segments

    def inpaint_face(self, empty_mask_segments):
        print("Start inpainting")
        empty_mask_segments = empty_mask_segments / 127.5 - 1
        inpaint_real = "inpaint_real"
        inpainted_faces = self.inpaint_model.predict(empty_mask_segments)
        print("Write results to dir: " + inpaint_real)
        for i in range(len(inpainted_faces)):
            inpainted_faces[i] = ((0.5 * inpainted_faces[i] + 0.5) * 255)
            cv2.imshow('img.jpg', (inpainted_faces[i]).astype("uint8"))
            k = cv2.waitKey(30) & 0xff
            cv2.imwrite(inpaint_real + "/" + str(i) + ".png", (inpainted_faces[i]).astype("uint8"))

    def merge(self, features, masks):
        print("Merge binary masked and faces")
        return dataset.merge(features, masks)


def capture_video(frames=30):
    print("capture video")
    cap = cv2.VideoCapture(0)
    images = []
    img_number = frames
    video_capture_dir = "video_capture"
    for i in range(img_number):
        ret, img = cap.read()
        cv2.imshow("video", img)
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break
        cv2.imwrite(video_capture_dir + "/" + str(i) + ".png", img.astype('uint8'))
        images.append(img)
    cap.release()
    cv2.destroyAllWindows()

    video_frames = np.stack(images)
    return video_frames


if __name__ == '__main__':
    inpaint_system = InpaintSystem()

    video_frames = capture_video(30)

    face_segments = inpaint_system.face_segment(video_frames)

    binary_segments = inpaint_system.binary_segmentation(face_segments)

    empty_mask_segments = inpaint_system.merge(face_segments, binary_segments)

    inpaint_system.inpaint_face(empty_mask_segments)

    face_recogntition_system.face_recognize()
