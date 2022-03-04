import os
from pathlib import Path
from PIL import Image
import numpy as np
import time
import logging

import torch
from torchvision import transforms, datasets

from .networks import ResnetEncoder, DepthDecoder
from .layers import disp_to_depth


class Monodepth2:
    
    def __init__ (self, model_name, use_cuda):
        self.logger = logging.getLogger('root.depth_estimator')
        self.device = "cuda" if use_cuda else "cpu"
        model_list = ['mono', 'mono+stereo', 'stereo']
        if model_name not in model_list:
            self.logger.fetal('monodepth2 models not exist')
            raise FileNotFoundError
        
        model_dir = Path(__file__).parent/'models'/model_name
        encoder_path = model_dir/'encoder.pth'
        decoder_path = model_dir/'depth.pth'
        
        self.logger.info('build encoder...')
        self.encoder = ResnetEncoder(18, False)
        loaded_dict_enc = torch.load(encoder_path, map_location=self.device)

        self.in_height = loaded_dict_enc['height']
        self.in_width = loaded_dict_enc['width']
        filtered_dict_enc = {k: v for k, v in loaded_dict_enc.items() if k in self.encoder.state_dict()}
        self.encoder.load_state_dict(filtered_dict_enc)
        
        self.logger.info('build decoder...')
        self.decoder = DepthDecoder(num_ch_enc=self.encoder.num_ch_enc, scales=range(4))
        loaded_dict = torch.load(decoder_path, map_location=self.device)
        self.decoder.load_state_dict(loaded_dict)
        
    def __call__ (self, ori_img, min_depth = 0.1, max_depth = 100):
        with torch.no_grad():
            input_img = Image.fromarray(ori_img)
            img_w, img_h = input_img.size
            preprocess = transforms.Compose([
                  transforms.Resize((self.in_height, self.in_width)),
                  transforms.ToTensor()])
            input_tensor = preprocess(input_img)
            input_batch = input_tensor.unsqueeze(0)
            features = self.encoder(input_batch)
            outputs = self.decoder(features)


            disp = outputs[("disp", 0)]
            _, depth = disp_to_depth(disp, min_depth, max_depth)
            depth_resized = torch.nn.functional.interpolate(depth, (img_h, img_w), mode="bilinear", align_corners=False)
            depth_resized_np = depth_resized.squeeze().cpu().numpy()
            return depth_resized_np
    
    def get_depth_center (self, depth_map, bbox_xywh):
        x,y,_,_ = bbox_xywh
        return depth_map[y][x]
    
    def get_depth_center_area (depth_map, tlwh, area = 2):
        pass
    
    def get_depth_bbaverage (depth_map, tlwh):
        pass
            