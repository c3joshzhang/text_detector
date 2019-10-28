#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 12:28:01 2019

@author: joshzhang
"""

class BaseTextDetector(object):
    def __init__(self, model_info=None, config_info=None):
        self.model_info = model_info 
        self.config_info = config_info
        self._model = {}
        self._config = {}
        self.build()

    def __call__(self, img):
        return self.detect(img)        
    
    def detect(self, img):
        print('--- Start Text Detecting ...')
        bounding_polys = self._detect(img)
        print('--- Text Detection Completed !!!')
        return bounding_polys
    
    def build(self):
        self._load_config(self.config_info)
        print('--- Model Configuration Loaded ...')
        print('    Configuration:', self._config or 'Default Args')
        self._load_model(self.model_info)
        print('--- Model is Ready ...')
            
    def reload_config(self, config_info):
        self._load_config(config_info)
        
    def reload_model(self, model_info):
        self._load_model(model_info)
        
    def _detect(self, img):
        raise NotImplementedError()
    
    def _load_model(self, model_info):
        raise NotImplementedError()
        
    def _load_config(self, config_info):
        raise NotImplementedError()
        