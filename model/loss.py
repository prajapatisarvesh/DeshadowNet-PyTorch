from typing import Any
import torch
import torch.nn as nn
from torch.nn import functional as F

class LogLoss(object):
    def __init__(self):
        super().__init__()
    
    def __call__(self, output, shadow, shadow_free):
        ### Convert to Log image
        output = (output * 255).to(dtype=torch.uint8)
        shadow = (shadow * 255).to(dtype=torch.uint8)
        shadow_free = (shadow_free * 255).to(dtype=torch.int)
        shadow_mask_log = torch.log10(shadow+1) - torch.log10(shadow_free+1)
        output_log = torch.log10(output+1)
        shadow_mask_log = shadow_mask_log.to(dtype=torch.float64)
        output_log = output_log.to(dtype=torch.float64)
        return F.mse_loss(output_log, shadow_mask_log)