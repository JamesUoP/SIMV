# -*- coding: utf-8 -*-
"""
Created on Tue Apr  5 15:58:03 2022

@author: Tommy
"""

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



image_path   = "./IMAGES/carTest.jpg"

yolo = Load_Yolo_model()

input_size=YOLO_INPUT_SIZE
original_image      = cv2.imread(image_path)
original_image      = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
original_image      = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)



image_data = image_preprocess(np.copy(original_image), [input_size, input_size])
image_data = image_data[np.newaxis, ...].astype(np.float32)


pred_bbox = yolo.predict(image_data)
show_confidence=True
score_threshold = 0.8
iou_threshold=0.45
CLASSES=YOLO_COCO_CLASSES
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
    print(label)




