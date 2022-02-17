import cv2
import numpy as np
from time import sleep
from PIL import Image

width_min = 20  # Minimum rectangle length
height_min = 20  # Minimum rectangle length

offset = 4  # Detection line offset

pos_line = 215  # Line position
line_left = 0
line_right = 250

detect = []
cars = 0
previous_frame = 0
previous_x = 0
fps = 80


def work_centre(x, y, w, h):
    x1 = int(w / 2)
    y1 = int(h / 2)
    cx = x + x1
    cy = y + y1
    return cx, cy


def screenshot(frame, x, y, w, h):
    im = Image.fromarray(frame, 'RGB')
    im1 = im.crop((x - w / 2, y - h / 2, x + w / 2, y + h / 2))
    im1.show()


def detection(detect, car_count):
    prev_frame = cap.get(1)
    prev_x = x
    if cap.get(1) > previous_frame + 20 or (x > previous_x + 50 or x < previous_x - 50):
        car_count += 1
        cv2.line(frame, (line_left, pos_line), (line_right, pos_line), (0, 127, 255), 1)
        print("car is detected : " + str(car_count) + " " + str(cap.get(1)))
        screenshot(frame, x, y, w, h)
    detect.remove((x, y))
    return prev_frame, prev_x, detect, car_count


kernel = None

cap = cv2.VideoCapture('4ktestvid_Trim10sec.mp4')
backgroundObject = cv2.createBackgroundSubtractorMOG2(detectShadows=True)

while True:
    ret, frame = cap.read()
    tempo = float(1 / fps)
    sleep(tempo)
    if not ret:
        break

    # Apply the background object on the frame to get the segement mask
    fgmask = backgroundObject.apply(frame)

    _, fgmask = cv2.threshold(fgmask, 250, 255, cv2.THRESH_BINARY)

    fgmask = cv2.erode(fgmask, kernel, iterations=1)
    fgmask = cv2.dilate(fgmask, kernel, iterations=2)

    contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    frameCopy = frame.copy()

    for cnt in contours:

        if cv2.contourArea(cnt) > 400:
            x, y, width, height = cv2.boundingRect(cnt)

            cv2.rectangle(frameCopy, (x, y), (x + width, y + height), (0, 0, 255), 2)

            cv2.putText(frameCopy, 'Car Detected', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1,
                        cv2.LINE_AA)

    cv2.line(frame, (line_left, pos_line), (line_right, pos_line), (255, 127, 0), 1)
    for (i, c) in enumerate(contours):
        (x, y, w, h) = cv2.boundingRect(c)
        validate_outline = (w >= width_min) and (h >= height_min)
        if not validate_outline:
            continue

        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        center = work_centre(x, y, w, h)
        detect.append(center)
        cv2.circle(frame, center, 4, (0, 0, 255), -1)

        for (x, y) in detect:
            if cars == 0:
                previous_x = x
            if ((pos_line + offset) > y > (pos_line - offset)) and (line_left < x < line_right):
                previous_frame, previous_x, detect, cars = detection(detect, cars)
                print(str(detect))

            if len(detect) == 10: #when detect is of size 10 we clear for storage reasons
                detect.clear()

    foregroundPart = cv2.bitwise_and(frame, frame, mask=fgmask)

    stacked = np.hstack((frame, foregroundPart, frameCopy))

    # cv2.imshow('Original Frame, Extracted Foreground and Detected Cars', cv2.resize(stacked, None, fx=0.5, fy=0.5))
    cv2.imshow('Original Frame, Extracted Foreground and Detected Cars', stacked)

    cv2.imshow('Clean Mask', fgmask)

    k = cv2.waitKey(1) & 0xff

    if k == ord('q'):
        break

cap.release()

cv2.destroyAllWindows()
