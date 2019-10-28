#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 23 14:22:40 2019

@author: joshzhang
"""
import cv2 as cv
import numpy as np
from skimage import io


def load_image_from_storage(img_storage):
    """
    Load image from data storage, package different methods to handle 
    different data storage types
    
    Args:
        img_storage (dict): {'type': str, 'location': str}
        
    Return:
        main_net (np.ndarray)
    """
    def _load_img_path(img_path):
        img = cv.imread(img_path)
        return img

    def _load_img_url(img_url):
        img = io.imread(img_url)
        return img
    img_load_func = {'path': _load_img_path, 'url': _load_img_url}
    img_typ = img_storage['type']
    img_loc = img_storage['location']
    img = img_load_func[img_typ](img_loc)
    return img

def combine_polygons(polys, img_size):
    '''
    Combine bounding polygons that are overlapping
    create empty image and fill polygons then use countors to detect the 
    overall region
    
    Args:
        polys (array of np.array): the polygons to be merged
        img_size: the reference img size which is the same as the model input 
    
    Return:
        combined (array of np.array(np.int32)): the merged polygons
    '''
    masked = np.zeros(img_size, dtype=np.uint8)
    for p in polys:
        masked = cv.fillPoly(masked, 
                              [np.array(p, dtype=np.int32)], 
                              color=2)
    ret, thresh = cv.threshold(masked, 1, 255, 0)
    a = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
    combined = a[0]
    return combined

def overlay_polys(img, boxes):
        """
        overlay the bounding boxes/polygons onto the original img

        Args:
            img (array): raw image context
            boxes (array): array of result file
                Shape: [num_detections, 4] for BB output
                / [num_detections, 4] for QUAD output

        Return:
            image (np.ndarray): with bounding polygons
        """
        img = np.array(img)
        for i, box in enumerate(boxes):
            poly = np.array(box).astype(np.int32).reshape((-1))
            poly = poly.reshape(-1, 2)
            cv.polylines(img, [poly.reshape((-1, 1, 2))],
                          True, color=(0, 0, 255),
                          thickness=2)
        return img


def remove_text(img, mask_polygons, method='polygon'):
    '''
    remove the text using the bouding polygons
    
    Args:
        img (np.ndarray): the orignal image
        mask_polygons (array(np.array)): polygon vertices array

    Return:
        removed (np.ndarray): a copy of image with text removed
    '''
    def _pixel_remove(img, polys):
        '''
        Assume that letters are black and the background is white
        avoid using this method if the diagram is not clean
        worse than poly_remove in time and space, 
        but can handle non-tight bounding boxes
        '''
        # fill non-text regions with white and create bool mask
        mask = np.zeros(img.shape[:2], np.uint8)
        mask = cv.fillPoly(mask, polys, True)
        non_text_mask = img.copy()
        non_text_mask[mask==0] = (255,255,255)
        non_text_mask = cv.cvtColor(non_text_mask, cv.COLOR_BGR2GRAY)
        # fill the text-region with white
        removed = img.copy()
        removed[non_text_mask<255] = (255,255,255)
        return removed
    
    def _poly_remove(img, polys):
        '''
        directly mask the bouding polygons with white and retur
        '''
        removed = img.copy()
        for mp in mask_polygons:
            removed = cv.fillPoly(removed, [mp], color=(255,255,255))
        return removed
    
    method_dict = {'pixel': _pixel_remove,'polygon': _poly_remove}
    if method not in method_dict:
        raise ValueError('only support method "pixel" or "polygon"')
    return method_dict[method](img, mask_polygons)