'''
LAST UPDATE: 2023.10.17
Course: CS7180
AUTHOR: Sarvesh Prajapati (SP), Abhinav Kumar (AK), Rupesh Pathak (RP)

E-MAIL: prajapati.s@northeastern.edu, kumar.abhina@northeastern.edu, pathal.r@northeastern.edu
DESCRIPTION: 
Lossfunction for deshadownet

'''
from typing import Any
import torch
import torch.nn as nn
from torch.nn import functional as F
import cv2

class LogLoss(object):
    def __init__(self):
        super().__init__()
    
    def __call__(self, output, shadow, shadow_free):
        ### Convert to Log image
        output = (output * 255).to(dtype=torch.uint8)
        shadow = (shadow * 255).to(dtype=torch.uint8)
        shadow_free = (shadow_free * 255).to(dtype=torch.uint8)
        shadow_mask_log = torch.log(shadow_free+1).clip(min=0, max=255) - torch.log(shadow+1)
        output_log = torch.log(output+1).clip(min=0, max=255)
        # x = shadow_mask_log.view(shadow_mask_log.shape[2], shadow_mask_log.shape[2], 3).to(device='cpu').numpy()
        # cv2.imshow('x', x)
        # cv2.waitKey(0)
        # shadow_mask_log = shadow_mask_log.to(dtype=torch.float64)
        # output_log = output_log.to(dtype=torch.float64)
        return F.mse_loss(output_log, shadow_mask_log)