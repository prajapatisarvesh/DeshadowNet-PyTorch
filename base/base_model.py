import numpy as np
import torch
import torch.nn as nn
from abc import abstractmethod

class BaseModel(nn.Module):
    '''
    BaseModel for other models initialization
    '''
    @abstractmethod
    def forward(self, *inputs):
        '''
        Implemented by other model
        '''
        raise NotImplementedError
    

    def __str__(self):
        '''
        Trainable Parameters
        '''
        model_params = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_params])
        return super().__str__() + '\nTrainable parameters: {}'.format(params)