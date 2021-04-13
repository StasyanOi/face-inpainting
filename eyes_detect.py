from __future__ import print_function
import cv2.cv2 as cv2

face_cascade_name = "haar/haarcascade_frontalface_alt.xml"
face_cascade = cv2.CascadeClassifier()
if not face_cascade.load(cv2.samples.findFile(face_cascade_name)):
    print('--(!)Error loading face cascade')
    exit(0)

def detectAndDisplay(frame):
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_gray = cv2.equalizeHist(frame_gray)
    faces = face_cascade.detectMultiScale(frame_gray)
    for (x, y, w, h) in faces:
        x_ = int(x - w / 3)
        y_ = int(y - h / 3) - 20
        x_w = int(x + w + w / 3)
        y_h = int(y + h + h / 3) - 20
        frame = frame[y_:y_h, x_:x_w]
    if frame is not None:
        frame = cv2.resize(frame, (256, 256))
        cv2.imwrite("test.png", frame)
        cv2.imshow('Capture - Face detection', frame)


cap = cv2.VideoCapture(0)
if not cap.isOpened:
    print('--(!)Error opening video capture')
    exit(0)
while True:
    ret, frame = cap.read()
    if frame is None:
        print('--(!) No captured frame -- Break!')
        break
    detectAndDisplay(frame)
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break
