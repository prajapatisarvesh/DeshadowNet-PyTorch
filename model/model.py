import torch
import torch.nn as nn
from torch.nn import functional as F
from base.base_model import BaseModel
import torchvision
from torchvision.models.feature_extraction import create_feature_extractor

class GNet(BaseModel):
    def __init__(self):
        super().__init__()
        self.required_layer = {
            'features.1':'relu1_1'
        }
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.vgg16_pretrained = torchvision.models.vgg16 (weights=torchvision.models.VGG16_Weights.IMAGENET1K_FEATURES)
        self.vgg16_pretrained.requires_grad_ = False
        '''
        Change Maxpool Stride to 1
        '''
        self.vgg16_pretrained.features[30] = nn.MaxPool2d(kernel_size=(1,1), stride=(1,1), padding=0, dilation=1, ceil_mode=False)
        self.vgg16_pretrained.features[23] = nn.MaxPool2d(kernel_size=(1,1), stride=(1,1), padding=0, dilation=1, ceil_mode=False)
        self.vgg16_pretrained.features.add_module('31', nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(1,1), stride=(1,1), padding=(0,0)))
        # self.vgg16_pretrained.features = nn.Sequential(*list(self.vgg16_pretrained.features.children())[0:24])
        '''
        Remove avgpool and classifier
        '''
        self.vgg16_pretrained = nn.Sequential(*list(self.vgg16_pretrained.children())[:-2])
        

    def forward(self, x):
        x = self.vgg16_pretrained(x)
        return x


class ANet(BaseModel):
    def __init__(self):
        super().__init__()
        self.prelu = nn.PReLU(num_parameters=64)
        self.dropout = nn.Dropout(p=0.2, inplace=True)
        self.conv1 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(5,5), stride=(1,1), padding=(2,2))
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(5,5), stride=(1,1), padding=(2,2))
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(5,5), stride=(1,1), padding=(2,2))
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(5,5), stride=(1,1), padding=(2,2))
        self.conv5 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(5,5), stride=(1,1), padding=(2,2))
        self.deconv1 = nn.ConvTranspose2d(in_channels=64, out_channels=3, kernel_size=(3,3), stride=(2,2), padding=(1,1))

    
    def forward(self, x):
        x = self.dropout(self.prelu(self.conv1(x)))
        x = self.dropout(self.prelu(self.conv2(x)))
        x = self.dropout(self.prelu(self.conv3(x)))
        x = self.dropout(self.prelu(self.conv4(x)))
        x = self.dropout(self.prelu(self.conv5(x)))
        return x


class DeShadowNet(BaseModel):
    def __init__(self):
        super().__init__()
        self.gnet = GNet()
        self.snet = ANet()
        self.anet = ANet()
    
    def forward(self, x):
        x = self.gnet(x)
        return x
