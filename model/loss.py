from typing import Any
import torch
import torch.nn as nn
from torch.nn import functional as F

class AlphaLoss(object):
    def __init__(self):
        super().__init__()

    
    def __call__(self, output_mask, target_mask):
        return F.mse_loss(output_mask, target_mask)


class CompositionLoss(object):
    def __init__(self):
        super().__init__()
        self.epsilon = 10e-5

    
    def __call__(self, output_shadow, target_shadow):
        return torch.sqrt(F.mse_loss(output_shadow, target_shadow)**2 + self.epsilon**2)


class CombinationLoss(object):
    def __init__(self):
        super().__init__()
        self.alpha_loss = AlphaLoss()
        self.composition_loss = CompositionLoss()
    
    def __call__(self, output, target_mask, shadow_free, shadow):
        return 0.8 * self.alpha_loss(output, target_mask) + 0.2 * self.composition_loss(shadow_free * output, shadow)