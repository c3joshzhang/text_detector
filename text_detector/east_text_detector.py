#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 23 14:19:10 2019

@author: joshzhang
"""
import numpy as np
import cv2 as cv
import math

from text_detector.text_detect_utils import combine_polygons
from text_detector.base_text_detector import BaseTextDetector
  
class EastTextDetector(BaseTextDetector):
    """
    Modified from:
    EAST: https://arxiv.org/abs/1704.03155v2
    https://github.com/opencv/opencv/blob/master/samples/dnn/text_detection.py
    """
    
    MAX_PIXEL_EAST = 134092800
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
          
    def _load_model(self, model_info):
        
        def _load_from_path(mdl_content):
            return cv.dnn.readNet(mdl_content['location'])   
            
        def _load_from_url(mdl_content):
            pass
        
        load_funcs = {'path': _load_from_path, 
                      'url':  _load_from_url}
        info_type = model_info['type']
        info_content = model_info['content']
        self._model['main'] = load_funcs[info_type](info_content) 
        
    
    def _load_config(self, config_info):
        if not config_info:
            return 
        
        def _load_from_dict(config_dict):
            return config_dict
            
        def _load_from_path(config_path):
            pass
             
        def _load_from_url(config_url):
            pass
        
        load_funcs = {'dict': _load_from_dict,
                      'path': _load_from_path,
                      'url':  _load_from_url}
        info_type = config_info['type']
        info_content = config_info['content']
        self._config = load_funcs[info_type](info_content)    
        
    def _detect(self, img):
        scaling = self._config.get('scaling', 1.0)
        confidence_threshold = self._config.get('confidence_threshold', 0.25)
        nms_threshold = self._config.get('nms_threshold', 0.8)
        
        # EAST args setup
        output_layers = ["feature_fusion/Conv_7/Sigmoid", 
                         "feature_fusion/concat_3"]
        pixel_scaling = 1.0
        img_shape = self.__find_shape(img, scaling)
        color_shift = (123.68, 116.78, 103.94)
        do_switch_rb = True
        do_crop_center = False
        blob = cv.dnn.blobFromImage(
                img, pixel_scaling, img_shape, 
                color_shift, 
                do_switch_rb,
                do_crop_center)
        # forward 
        self._model['main'].setInput(blob)
        scores, geometry = self._model['main'].forward(output_layers)
        bboxes, rectrangles, confidences = \
                self.__decode(img, scores, geometry, 
                              confidence_threshold,
                              *img_shape)
        # merge
        indicies = cv.dnn.NMSBoxesRotated(
                bboxes, confidences, 
                confidence_threshold, 
                nms_threshold)
        boundings = []
        for idx in indicies:
            r = rectrangles[idx[0]]
            if r[1][0] < r[0][0] and r[0][1] < r[1][1]:
                top_left = r[0]
                top_right = [r[0][0], r[1][1]]
                bottom_right = r[1] 
                bottom_left = [r[1][0], r[0][1]]
                b = np.array([top_left, top_right, bottom_right, bottom_left],
                             dtype=np.int32)
                boundings.append(b)
        combined = combine_polygons(boundings, img.shape[:2])
        return combined
            
    def __find_shape(self, img, scaling):
        myround = lambda x,base: base * round(x/base)
        h, w, _ = img.shape
        h, w = h*scaling, w*scaling
        if (h * w) > self.MAX_PIXEL_EAST:
            ratio = h / w
            w = (self.MAX_PIXEL_EAST / ratio) ** 0.5
            h = w * ratio
        return int(myround(w,32)), int(myround(h,32))
    
    
    def __decode(self, img, scores, geometry, score_thresh, W, H):
        detections = []
        confidences = []
        rectangles = []
    
        # CHECK DIMENSIONS AND SHAPES OF geometry AND scores #
        assert len(scores.shape) == 4, "Incorrect dimensions of scores"
        assert len(geometry.shape) == 4, "Incorrect dimensions of geometry"
        assert scores.shape[0] == 1, "Invalid dimensions of scores"
        assert geometry.shape[0] == 1, "Invalid dimensions of geometry"
        assert scores.shape[1] == 1, "Invalid dimensions of scores"
        assert geometry.shape[1] == 5, "Invalid dimensions of geometry"
        assert scores.shape[2] == geometry.shape[2], "Invalid dimensions of scores and geometry"
        assert scores.shape[3] == geometry.shape[3], "Invalid dimensions of scores and geometry"
        height = scores.shape[2]
        width = scores.shape[3]
    
        def _get_rectangle(img, p1, p3):
            img_height, img_length = img.shape[:2]
            ratio_height, ratio_length = img_height / H, img_length / W
            p2 = (int(p1[0] * ratio_length), int(p1[1] * ratio_height))
            p4 = (int(p3[0] * ratio_length), int(p3[1] * ratio_height))
            return p2, p4
    
        for y in range(0, height):
    
            # Extract data from scores
            scoresData = scores[0][0][y]
            x0_data = geometry[0][0][y]
            x1_data = geometry[0][1][y]
            x2_data = geometry[0][2][y]
            x3_data = geometry[0][3][y]
            anglesData = geometry[0][4][y]
            for x in range(0, width):
                score = scoresData[x]
    
                # If score is lower than threshold score, move to next x
                if score < score_thresh:
                    continue
    
                # Calculate offset
                offsetX = x * 4.0
                offsetY = y * 4.0
                angle = anglesData[x]
    
                # Calculate cos and sin of angle
                cosA = math.cos(angle)
                sinA = math.sin(angle)
                h = x0_data[x] + x2_data[x]
                w = x1_data[x] + x3_data[x]
    
                # Calculate offset
                offset = ([
                    offsetX + cosA * x1_data[x] + sinA * x2_data[x],
                    offsetY - sinA * x1_data[x] + cosA * x2_data[x]
                ])
    
                # Find points for rectangle
                p1 = (-sinA * h + offset[0], -cosA * h + offset[1])
                p3 = (-cosA * w + offset[0], sinA * w + offset[1])
                center = (0.5 * (p1[0] + p3[0]), 0.5 * (p1[1] + p3[1]))
    
                detections.append((center, (w, h), -1 * angle * 180.0 / math.pi))
                rectangles.append(_get_rectangle(img, p1, p3))
                confidences.append(float(score))
    
        # Return detections and confidences
        return [detections, rectangles, confidences]
            
    
