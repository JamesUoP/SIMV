import cv2
import numpy as np
from time import sleep

width_min = 50 
height_min = 50

offset = 3  
pos_line = 220
delay = 60
detect = []
cars = 0


def work_centre(x, y, w, h):
    x1 = int(w / 2)
    y1 = int(h / 2)
    cx = x + x1
    cy = y + y1
    return cx, cy


cap = cv2.VideoCapture('superbusyvid.mp4')
subtractor = cv2.bgsegm.createBackgroundSubtractorMOG()

while True:
    ret, frame1 = cap.read()
    tempo = float(1 / delay)
    sleep(tempo)
    grey = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(grey, (3, 3), 5)
    img_sub = subtractor.apply(blur)
    expand = cv2.dilate(img_sub, np.ones((5, 5)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    expanded = cv2.morphologyEx(expand, cv2.MORPH_CLOSE, kernel)
    expanded = cv2.morphologyEx(expanded, cv2.MORPH_CLOSE, kernel)
    outline, h = cv2.findContours(expanded, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    cv2.line(frame1, (25, pos_line), (1200, pos_line), (255, 127, 0), 3)
    for (i, c) in enumerate(outline):
        (x, y, w, h) = cv2.boundingRect(c)
        validate_outline = (w >= width_min) and (h >= height_min)
        if not validate_outline:
            continue

        cv2.rectangle(frame1, (x, y), (x + w, y + h), (0, 255, 0), 2)
        center = work_centre(x, y, w, h)
        detect.append(center)
        cv2.circle(frame1, center, 4, (0, 0, 255), -1)

        for (x, y) in detect:
            if (pos_line + offset) > y > (pos_line - offset):
                cars += 1
                cv2.line(frame1, (25, pos_line), (1200, pos_line), (0, 127, 255), 3)
                detect.remove((x, y))
                print("car is detected : " + str(cars))
                print(x, y, w, h)


    cv2.putText(frame1, "VEHICLE COUNT : " + str(cars), (450, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 5)
    cv2.imshow("Original Video", frame1)
    cv2.imshow("Detector", expanded)

    if cv2.waitKey(1) == 27:
        break

cv2.destroyAllWindows()
cap.release()
