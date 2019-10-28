#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 23 17:53:01 2019

@author: joshzhang
"""
from text_detector.third_party.CRAFT.craft import CRAFT
from text_detector.third_party.CRAFT.refinenet import RefineNet
from text_detector.third_party.CRAFT import craft_utils
from text_detector.third_party.CRAFT import imgproc
from text_detector.text_detect_utils import combine_polygons
from text_detector.base_text_detector import BaseTextDetector

import urllib.request 
import numpy as np
import cv2

import torch
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from collections import OrderedDict

class CraftTextDetector(BaseTextDetector):
    """
    Text Detector and extractor
    Modified from CRAFT: https://github.com/clovaai/CRAFT-pytorch
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def _detect(self, img):
        """
        Detect all the texts by cropping the image to reduce memory requirement 
        by trading off speed, the memory requirement increases polynominally 
    
        Args:
            img (np.ndarray): the image tensor
            main_net (torch net): main CRAFT net
            refine_net (torch net): link refiner net
            sub_img_h (unsigned int): the cropped image height
            sub_img_w (unsigned int): the cropped image width
            sub_img_overlap (unsigned int): the size of the gap filler
            verbose (bool): if showing the step progress
    
        Return:
            combined_polys (array[np.array]): the text bouding polygons
        """
        
        sub_img_h = self._config.get('sub_img_h', 1080)
        sub_img_w = self._config.get('sub_img_w', 1080)
        sub_img_overlap = self._config.get('sub_img_overlap', 360)
        mag_ratio = self._config.get('mag_ratio', 2)
        text_threshold = self._config.get('text_threshold', 0.7)
        link_threshold = self._config.get('link_threshold', 0.7)
        low_text = self._config.get('low_text', 0.4)
        cuda = self._config.get('cuda', False)
        poly = self._config.get('poly', True)
        main_net = self._model.get('main_net', None)
        refine_net = self._model.get('refine_net', None)
        
        full_h, full_w = img.shape[0], img.shape[1]
        vert_steps = int(full_h/sub_img_h) + 1
        hori_steps = int(full_w/sub_img_w) + 1
        all_polys = []
        for vert_step in range(vert_steps):
            for hori_step in range(hori_steps):
                print('vertical step: {}/{}, horizontal step: {}/{}'\
                      .format(vert_step+1, vert_steps,
                              hori_step+1, hori_steps))
                
                # main sub crop
                x_start = vert_step * sub_img_h
                x_end = min(x_start + sub_img_h, full_h)
                y_start = hori_step * sub_img_w
                y_end = min(y_start + sub_img_w, full_w)
                main_crop = img[x_start: x_end, y_start: y_end, :]
                _, main_polys = self.__craft_net(main_net, refine_net,
                                                 main_crop, 
                                                 x_end - x_start,
                                                 mag_ratio,
                                                 text_threshold,
                                                 link_threshold,
                                                 low_text,
                                                 cuda,
                                                poly)
                for p in main_polys:
                    all_polys.append(p + np.array([y_start, x_start]))
                    
                # vertical gap filler
                vert_x_start = x_end - int(sub_img_overlap/2)
                vert_x_end = min(vert_x_start + sub_img_overlap, full_h)
                vert_crop = img[vert_x_start: vert_x_end, y_start: y_end, :]
                _, vert_polys = self.__craft_net(main_net, refine_net,
                                                 vert_crop, 
                                                 vert_x_end - vert_x_start,
                                                 mag_ratio,
                                                 text_threshold,
                                                 link_threshold,
                                                 low_text,
                                                 cuda,
                                                 poly)
                for p in vert_polys:
                    all_polys.append(p + np.array([y_start, vert_x_start]))
                    
                # horizontal gap filler
                hori_y_start = y_end - int(sub_img_overlap/2)
                hori_y_end = min(hori_y_start + sub_img_overlap, full_w)
                hori_crop = img[x_start: x_end, hori_y_start: hori_y_end, :]
                _, hori_polys = self.__craft_net(main_net, refine_net,
                                                 hori_crop, 
                                                 x_end - x_start,
                                                 mag_ratio,
                                                 text_threshold,
                                                 link_threshold,
                                                 low_text,
                                                 cuda,
                                                 poly)
                for p in hori_polys:
                    all_polys.append(p + np.array([hori_y_start, x_start]))
        # merge all overlapping polygons 
        combined_polys = combine_polygons(all_polys, img.shape[:-1])
        return combined_polys
        
    def _load_model(self, model_info):
        """
        Load the models for the detector
        
        Args:
            model_info (dict): {'type': ..., 'content': ...} model dump info
            
        Return:
            None, the protected class variable _model is set 
        """
        main_net_info = model_info['main_net']
        refine_net_info = model_info['refine_net']
        self._model['main_net'] = self.__load_main_net_from_info(main_net_info)
        self._model['refine_net'] = self.__load_refine_net_from_info(refine_net_info)
    
    def _load_config(self, config_info):
        """
        Load the model forward parameters and the detection parameters
        
        Args:
            config_info (dict): {'type': ..., 'content': ...} detector params
        
        Return:
            None, the protected class variable _config is set
        """
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
    
    def __craft_net(self, main_net, refine_net,
                    image, canvas_size,
                    mag_ratio=2,
                    text_threshold=0.7,
                    link_threshold=0.4,
                    low_text=0.4,
                    cuda=False,
                    poly=True):
        """
        apply main CRAFT net and the link refiner net to generate text bounding
        polygons
        
        Args:
            main_net: trained torch net object
            refine_net: trained torch net object
            image (np.ndarray): single image 
            canvas_size (unsigned int): the image size to resiz to 
            mag_ratio (float): the ratio to magnified 
            text_threshold (float): text confidence threshold
            link_threshold (float): link confidence threshold
            low_text (float): text low-bound score
            cuda (bool): if use cuda
            poly (bool): if return polygons
            
        Return:
            bboxes: bounding boxes
            polys: bounding polygons 
            ret_score_text: detected text scores
        """
        # resize
        img_resized, target_ratio, size_heatmap = \
            imgproc.resize_aspect_ratio(image, canvas_size, 
                                        interpolation=cv2.INTER_LINEAR, 
                                        mag_ratio=mag_ratio)
        ratio_h = ratio_w = 1 / target_ratio
        # preprocessing
        x = imgproc.normalizeMeanVariance(img_resized)
        x = torch.from_numpy(x).permute(2, 0, 1)    # [h, w, c] to [c, h, w]
        x = Variable(x.unsqueeze(0))                # [c, h, w] to [b, c, h, w]
        if cuda:
            x = x.cuda()
        # forward pass
        y, feature = main_net(x)
        # make score and link map
        score_text = y[0,:,:,0].cpu().data.numpy()
        score_link = y[0,:,:,1].cpu().data.numpy()
        # refine link
        if refine_net is not None:
            y_refiner = refine_net(y, feature)
            score_link = y_refiner[0,:,:,0].cpu().data.numpy()
        # Post-processing
        boxes, polys = craft_utils\
            .getDetBoxes(score_text, score_link, 
                         text_threshold, link_threshold, 
                         low_text, poly)
        # coordinate adjustment
        boxes = craft_utils.adjustResultCoordinates(boxes, ratio_w, ratio_h)
        polys = craft_utils.adjustResultCoordinates(polys, ratio_w, ratio_h)
        for k in range(len(polys)):
            if polys[k] is None: polys[k] = boxes[k]
        return boxes, polys
            
    def __load_main_net_from_info(self, main_net_info):
        """
        Load main net from data storage, package different methods to handle 
        different data storage types
        
        Args:
            main_net_storage (dict): {'type': str, 'location': str}
            
        Return:
            main_net (CraftNet object)
        """
        def _load_main_net_path(mn_cnt):
            main_net_path = mn_cnt['location']
            main_net = self.__load_craft_main(main_net_path)
            return main_net
        
        def _load_main_net_url(mn_cnt):
            local_net_path = "craft_main.pth"
            try:
                with open(local_net_path):
                    pass
            except IOError:
                main_net_url = mn_cnt['location']
                print('Downloading craft net main ...')
                urllib.request.urlretrieve(main_net_url, local_net_path)
            main_net = self.__load_craft_main(local_net_path)
            return main_net
        
        img_main_func = {'path': _load_main_net_path, 
                         'url': _load_main_net_url}
        mn_typ = main_net_info['type']
        mn_cnt = main_net_info['content']
        main_net = img_main_func[mn_typ](mn_cnt)
        return main_net

    def __load_refine_net_from_info(self, refine_net_info):  
        """
        Load refine net from data storage, package different methods to handle 
        different data storage types
        
        Args:
            refine_net_storage (dict): {'type': str, 'location': str}
            
        Return:
            refine_net (RefineNet object)
        """
        def _load_refine_net_path(rn_cnt):
                refine_net_path = rn_cnt['location']
                refine_net = self.__load_craft_refine(refine_net_path)
                return refine_net
            
        def _load_refine_net_url(rn_cnt):
                local_net_path = "craft_refine.pth"
                try:
                    with open(local_net_path):
                        pass
                except IOError:
                    print('Downloading craft net refiner ...')
                    refine_net_url = rn_cnt['location']
                    urllib.request.urlretrieve(refine_net_url, local_net_path)
                refine_net = self.__load_craft_refine(local_net_path)
                return refine_net
            
        img_refine_func = {'path': _load_refine_net_path, 
                           'url': _load_refine_net_url}
        rn_typ = refine_net_info['type']
        rn_cnt = refine_net_info['content']
        refine_net = img_refine_func[rn_typ](rn_cnt)
        return refine_net
            
    def __load_craft_main(self, main_mdl_path, cuda=False):
        '''Load the main CRAFT text localization net
    
        Args:
            main_mdl_path (str): path for the model dump to be loaded
            cuda (bool): if using cuda 
        Return:
            None
        '''
        net = CRAFT() 
        if cuda:
            net.load_state_dict(
                self.__copyStateDict(torch.load(main_mdl_path)))
        else:
            net.load_state_dict(
                self.__copyStateDict(torch.load(main_mdl_path, 
                                                map_location='cpu')))
        if cuda:
            net = net.cuda()
            net = torch.nn.DataParallel(net)
            cudnn.benchmark = False
        net.eval()
        return net
    
    def __load_craft_refine(self, refine_mdl_path, cuda=False):
        '''
        Load CRAFT refiner net, a helper net to refine the linkage
        
        Args:
            refine_mdl_path (str): refiner net dump to be loaded
            cuda (bool): if using cuda
        Return:
            None
        '''
        refine_net = RefineNet()
        if cuda:
            refine_net.load_state_dict(
                self.__copyStateDict(torch.load(refine_mdl_path)))
            refine_net = refine_net.cuda()
            refine_net = torch.nn.DataParallel(refine_net)
        else:
            refine_net.load_state_dict(
                self.__copyStateDict(
                    torch.load(refine_mdl_path, map_location='cpu')))
        refine_net.eval()
        return refine_net
    
    def __copyStateDict(self, state_dict):
        """
        Help function to load the torch net
    
        Args:
            state_dict: loaded object from torch.load
        Return:
            new_state_dict: formatted dict for torch 
        """
        if list(state_dict.keys())[0].startswith("module"):
            start_idx = 1
        else:
            start_idx = 0
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = ".".join(k.split(".")[start_idx:])
            new_state_dict[name] = v
        return new_state_dict