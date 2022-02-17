import cv2
import numpy as np
from time import sleep
from PIL import Image
fps = 60

kernel = None

cap = cv2.VideoCapture('4ktestvid_Trim10sec.mp4')
backgroundObject = cv2.createBackgroundSubtractorMOG2(detectShadows=True)

while True:
    ret, frame = cap.read()
    tempo = float(1 / fps)
    sleep(tempo)
    if not ret:
        break

    #Apply the background object on the frame to get the segement mask
    fgmask = backgroundObject.apply(frame)

    _, fgmask = cv2.threshold(fgmask, 250, 255, cv2.THRESH_BINARY)

    fgmask = cv2.erode(fgmask, kernel, iterations = 1)
    fgmask = cv2.dilate(fgmask, kernel, iterations = 2)


    contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    frameCopy = frame.copy()

    for cnt in contours:

        if cv2.contourArea(cnt) > 400:
            x, y, width, height = cv2.boundingRect(cnt)

            cv2.rectangle(frameCopy, (x , y), (x + width , y+ height), (0, 0, 255), 2)

            cv2.putText(frameCopy, 'Car Detected', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1, cv2.LINE_AA)

    foregroundPart = cv2.bitwise_and(frame, frame, mask=fgmask)

    stacked = np.hstack((frame, foregroundPart, frameCopy))

    #cv2.imshow('Original Frame, Extracted Foreground and Detected Cars', cv2.resize(stacked, None, fx=0.5, fy=0.5))
    cv2.imshow('Original Frame, Extracted Foreground and Detected Cars', stacked)

    cv2.imshow('Clean Mask', fgmask)

    k = cv2.waitKey(1) & 0xff

    if k == ord('q'):
        break

cap.release()

cv2.destroyAllWindows()