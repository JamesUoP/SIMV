# -*- coding: utf-8 -*-
"""
Created on Fri Apr  1 12:55:48 2022

@author: Tommy
"""

import time
import cv2
import numpy as np
from time import sleep
from PIL import Image
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import cv2
import numpy as np
import tensorflow as tf
from yolov3.utils import detect_image, detect_realtime, detect_video, Load_Yolo_model, detect_video_realtime_mp, postprocess_boxes, nms, image_preprocess
from yolov3.configs import *
from multiprocessing import Process, Queue, Pipe
import cv2
import time
import random
import colorsys
import numpy as np
import tensorflow as tf
from yolov3.configs import *
from yolov3.yolov4 import *
from tensorflow.python.saved_model import tag_constants
#----------------
yolo = Load_Yolo_model()
#----------------

test_mode = False
test_mode_screenshot = True
test_mode_bounding_boxes = False
lineWidth = 1
width_min = 40  # Minimum rectangle length
height_min = 40  # Minimum rectangle length

offset = 4  # Detection line offset
reset_time = 10
# pos_line = 215  # Line position
# line_left = 0
# line_right = 250

fps = 100


class Detector:

    def __init__(self, videos, line_position, results):
        self.videos = videos
        self.results = results
        self.out_pos_line, self.in_pos_line = line_position[0], line_position[3]
        self.out_line_left, self.in_line_left = line_position[1], line_position[4]
        self.out_line_right, self.in_line_right = line_position[2], line_position[5]

    def vehicle_detection(self, id):
        cap = cv2.VideoCapture(self.videos[id])
        out_pos_line, in_pos_line = self.out_pos_line[id], self.in_pos_line[id]
        out_line_left, in_line_left = self.out_line_left[id], self.in_line_left[id]
        out_line_right, in_line_right = self.out_line_right[id], self.in_line_right[id]
        kernel = None
        running = True
        detect = []
        cars_in = 0
        cars_out = 0
        previous_frame = 0
        previous_x = 0
        backgroundObject = cv2.createBackgroundSubtractorMOG2(detectShadows=True)
        current_time = time.time()

        while running:
            ret, frame = cap.read()
            tempo = float(1 / fps)
            if test_mode:
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
                    if test_mode_bounding_boxes:
                        cv2.rectangle(frameCopy, (x, y), (x + width, y + height), (0, 0, 255), lineWidth)

                    cv2.putText(frameCopy, 'Car Detected', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1,
                                cv2.LINE_AA)
                if test_mode_bounding_boxes:
                    cv2.line(frame, (out_line_left, out_pos_line), (out_line_right, out_pos_line), (255, 127, 0), lineWidth)
                    cv2.line(frame, (in_line_left, in_pos_line), (in_line_right, in_pos_line), (255, 127, 0), lineWidth)
                    
            for (i, c) in enumerate(contours):
                (x, y, w, h) = cv2.boundingRect(c)
                validate_outline = (w >= width_min) and (h >= height_min)
                if not validate_outline:
                    continue
                if test_mode_bounding_boxes:
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), lineWidth)
                center = work_centre(x, y, w, h)
                detect.append(center)
                if test_mode_bounding_boxes:
                    cv2.circle(frame, center, 4, (0, 0, 255), -lineWidth)

                for (x, y) in detect:
                    if cars_in and cars_out == 0:
                        previous_x = x

                    if ((out_pos_line + offset) > y > (out_pos_line - offset)) and (out_line_left < x < out_line_right):
                        previous_frame, previous_x, detect, cars_out = self.detection(cap, detect, cars_out,
                                                                                      frame, x,
                                                                                      y, w, h,
                                                                                      previous_frame, previous_x, id,
                                                                                      is_in=False)

                    if ((in_pos_line + offset) > y > (in_pos_line - offset)) and (
                            in_line_left < x < in_line_right):
                        previous_frame, previous_x, detect, cars_in = self.detection(cap, detect, cars_in,
                                                                                     frame, x, y,
                                                                                     w, h,
                                                                                     previous_frame, previous_x, id,
                                                                                     is_in=True)
                        # print(str(detect)) Prints list of bounding boxes

                    if len(detect) == 10:  # when detect is of size 10 we clear for storage reasons
                        detect.clear()

                    self.results[id] = [cars_out, cars_in]

            foregroundPart = cv2.bitwise_and(frame, frame, mask=fgmask)

            stacked = np.hstack((frame, foregroundPart, frameCopy))

            if test_mode:
                # cv2.imshow('Original Frame, Extracted Foreground and Detected Cars', cv2.resize(stacked, None, fx=0.5, fy=0.5))
                cv2.imshow('video{} Stacked'.format(id), stacked)

                # cv2.imshow('video{} Clean Mask'.format(id), fgmask)

            k = cv2.waitKey(1) & 0xff

            if time.time() > current_time + reset_time:
                running = False

            if k == ord('q'):
                break
        cap.release()
        if id == 0:
            print("cars in: " + str(cars_in) + " cars out: " + str(cars_out))
        # cv2.destroyAllWindows()
        #self.results[id] = [cars_out, cars_in]

    def detection(self, cap, detect, car_count, frame, x, y, w, h, previous_frame, previous_x, id, is_in):
        prev_frame = cap.get(1)
        prev_x = x
        if cap.get(1) > previous_frame + 20 or (x > previous_x + 50 or x < previous_x - 50):
            car_count += 1
            if is_in:
                if test_mode_bounding_boxes:
                    cv2.line(frame, (self.in_line_left[id], self.in_pos_line[id]),
                         (self.in_line_right[id], self.in_pos_line[id]),
                         (0, 127, 255), 1)
                if id == 0:
                    print(" IN: " + str(car_count) + " " + str(cap.get(1)))
            else:
                if test_mode_bounding_boxes:
                    cv2.line(frame, (self.out_line_left[id], self.out_pos_line[id]),
                         (self.out_line_right[id], self.out_pos_line[id]),
                         (0, 127, 255), 1)
                if id == 0:
                    print(" OUT: " + str(car_count) + " " + str(cap.get(1)))
            screenshot(frame, x, y, w, h)
        detect.remove((x, y))
        return prev_frame, prev_x, detect, car_count

    def output(self):
        return self.results


def work_centre(x, y, w, h):
    x1 = int(w / 2)
    y1 = int(h / 2)
    cx = x + x1
    cy = y + y1
    return cx, cy


def screenshot(frame, x, y, w, h):
    im = Image.fromarray(frame, 'RGB')
    w = w+10
    h = h+10
    im1 = im.crop((x - w / 2, y - h / 2, x + w / 2, y + h / 2))
    
    input_size=YOLO_INPUT_SIZE
    
    
    
    original_image = im1
    original_image = np.array(original_image)
    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    
    
    
    image_data = image_preprocess(np.copy(original_image), [input_size, input_size])
    image_data = image_data[np.newaxis, ...].astype(np.float32)
    
    
    pred_bbox = yolo.predict(image_data)
    show_confidence=True
    score_threshold = 0.4
    iou_threshold=0.45
    
    #----------------------------
    CLASSES=YOLO_COCO_CLASSES
    #----------------------------
    
    NUM_CLASS = read_class_names(CLASSES)
    
    pred_bbox = [tf.reshape(x, (-1, tf.shape(x)[-1])) for x in pred_bbox]
    pred_bbox = tf.concat(pred_bbox, axis=0)
    bboxes = postprocess_boxes(pred_bbox, original_image, input_size, score_threshold)
    bboxes = nms(bboxes, iou_threshold, method='nms')
    
    for i, bbox in enumerate(bboxes):
        coor = np.array(bbox[:4], dtype=np.int32)
        score = bbox[4]
        class_ind = int(bbox[5])
            # get text label
        score_str = " {:.2f}".format(score) if show_confidence else ""
    
        try:
            label = "{}".format(NUM_CLASS[class_ind]) + score_str
        except KeyError:
            print("You received KeyError, this might be that you are trying to use yolo original weights")
            print("while using custom classes, if using custom model in configs.py set YOLO_CUSTOM_WEIGHTS = True")
        print("detected",label)
    if test_mode_screenshot:
        im1.show('video{} Scrn'.format(id))