"""
Butterfly tracking tool - SIPLab

Full_Butterfly_Detection_Module.py
Created: 2020/06/22
Author: Alejandro Solís Hernández

Description:
    Module to detect monarch butterflies in monochromic images.
    This detector uses darknet (Tiny YOLOv3), by installing the yolopy34 module
"""

import os
import cv2
import sys
import subprocess
import numpy as np
import torch

from pydarknet import Detector, Image

CONFIDENCE = 0.5

DIR_PATH = os.path.dirname(os.path.realpath(__file__))

OBJ_INFO      =  DIR_PATH + "/data/obj.data"
CFG_FILE      =  DIR_PATH + "/cfg/butterfly-tiny-yolov3-3500.cfg"
WEIGHTS_FILE  =  DIR_PATH + "/weights/butterfly-tiny-yolov3-3500_170000.weights"

# Create object metadata
subprocess.call("./create_obj_info.sh", cwd=DIR_PATH)

class FullButterflyDetector:
    def __init__(self, confidence=CONFIDENCE):
        self.confidence = confidence
        self.net = Detector(bytes(CFG_FILE, encoding="utf-8"), 
                            bytes(WEIGHTS_FILE, encoding="utf-8"), 
                            0, 
                            bytes(OBJ_INFO,encoding="utf-8")
                           )
        self.progress = 0

    def convert_xy(self, pts):
        x = pts[0]
        y = pts[1]
        w = pts[2]
        h = pts[3]
        
        x1 = x
        y1 = y
        x2 = x+w
        y2 = y+h

        return [x1, y1, x2, y2]

    def detect_img(self, img):
        detections = None
        img_info = None

        img_darknet = Image(img)
        h,w = img.shape[:2]

        img_info = {'id': 0, 'file_name': None, 'height': h, 'width': w, 'raw_img': img, 'ratio': 1.111111111}

        results = self.net.detect(img_darknet, self.confidence)

        dets = []
        for tag in results:
            label = self.convert_xy(tag[2])
            dets += [label + [tag[1]] + [tag[1]] + [0.0]]

        #print(dets)
        dets = np.asarray(dets)
        #print(dets)
        detections = [torch.tensor(np.array(dets), device='cuda')]

        '''
        <class 'torch.Tensor'>
        tensor([[ 6.3850e+02,  2.9225e+02,  8.9350e+02,  9.2300e+02,  9.9854e-01,
          9.2725e-01,  0.0000e+00],
        [ 1.1010e+03,  7.2750e+02,  1.3310e+03,  1.1340e+03,  2.8458e-03,
          4.0454e-01,  0.0000e+00],
        [ 6.1800e+02,  3.7000e+02,  6.4200e+02,  4.2900e+02,  1.5974e-03,
          7.0996e-01,  0.0000e+00],
        [-5.3800e+02,  2.4000e+02,  8.7750e+01,  1.6770e+03,  2.0752e-03,
          5.2051e-01,  0.0000e+00],
        [ 4.3800e+02, -7.3800e+02,  8.5900e+02,  2.3438e+02,  1.6546e-03,
          6.2061e-01,  0.0000e+00],
        [-5.3850e+02,  2.1050e+02,  2.2338e+02,  1.9960e+03,  2.1992e-03,
          4.6191e-01,  0.0000e+00]], device='cuda:0')
        7
        '''

        return detections, img_info

    def detect(self, imagesList):

        totalImages = len(imagesList)
        self.progress = 0
        detections = []
        for i, imageFilename in enumerate(imagesList):
            img_detections = []

            img = cv2.imread(imageFilename)
            img_darknet = Image(img)
            h,w = img.shape[:2]

            result = self.net.detect(img_darknet, self.confidence)
            for cat, score, bounds in result:
                x, y, dx, dy = bounds
                nx, ny, ndx, ndy = x/w, y/h, dx/w, dy/h
                img_detections.append([nx,ny,ndx,ndy])

            detections.append(img_detections)
            self.progress = (i+1)/totalImages

        return detections

    def get_progress(self):
        return self.progress

