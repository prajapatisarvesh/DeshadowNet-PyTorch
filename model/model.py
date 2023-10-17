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
        self.vgg16_pretrained.features.add_module('32', nn.ReLU(inplace=True))
        # self.vgg16_pretrained.features = nn.Sequential(*list(self.vgg16_pretrained.features.children())[0:24])
        '''
        Remove avgpool and classifier
        '''
        self.vgg16_pretrained = nn.Sequential(*list(self.vgg16_pretrained.children())[:-2])
        

    def forward(self, x):
        x = self.vgg16_pretrained(x)
        return x.clip(min=0, max=1)


class ANet(BaseModel):
    def __init__(self):
        super().__init__()
        self.prelu = nn.PReLU(num_parameters=64)
        self.dropout = nn.Dropout(p=0.6, inplace=True)
        self.conv1 = nn.Conv2d(in_channels=160, out_channels=64, kernel_size=(5,5), stride=(1,1), padding=(2,2))
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(5,5), stride=(1,1), padding=(2,2))
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(5,5), stride=(1,1), padding=(2,2))
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(5,5), stride=(1,1), padding=(2,2))
        self.deconv1 = nn.ConvTranspose2d(in_channels=64, out_channels=3, kernel_size=(4,4), stride=(2,2), padding=(1,1))

    
    def forward(self, x):
        x = self.dropout(self.prelu(self.conv1(x)))
        x = self.dropout(self.prelu(self.conv2(x)))
        x = self.dropout(self.prelu(self.conv3(x)))
        x = self.dropout(self.prelu(self.conv4(x)))
        x = self.deconv1(x)
        return x.clip(min=0, max=1)


class DeShadowNet(BaseModel):
    def __init__(self):
        super().__init__()
        self.gnet = GNet()
        self.snet = ANet()
        self.anet = ANet()
        self.dropout = nn.Dropout(p=0.6)
        self.deconv1 = nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=(8,8), stride=(4,4), padding=(2,2))
        self.deconv2 = nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=(8,8), stride=(4,4), padding=(2,2))
        self.conv21 = nn.Conv2d(in_channels=3, out_channels=96, kernel_size=(9,9), stride=(1,1), padding=(4,4))
        self.conv31 = nn.Conv2d(in_channels=3, out_channels=96, kernel_size=(9,9), stride=(1,1), padding=(4,4))
        self.maxpool21 = nn.MaxPool2d(kernel_size=(3,3), stride=(2,2),padding=1, dilation=1, ceil_mode=False)
        self.maxpool22 = nn.MaxPool2d(kernel_size=(3,3), stride=(2,2),padding=1, dilation=1, ceil_mode=False)
        self.conv22 = nn.Conv2d(in_channels=256, out_channels=64, kernel_size=(1,1), stride=(1,1), padding=(0,0))
        self.conv32 = nn.Conv2d(in_channels=256, out_channels=64, kernel_size=(1,1), stride=(1,1), padding=(0,0))
        self.final_conv = nn.Conv2d(in_channels=6, out_channels=3, kernel_size=(1,1), stride=(1,1), padding=(0,0))


    def forward(self, x):
        c11 = self.maxpool21(self.dropout(F.relu(self.conv21(x), inplace=True)))
        c21 = self.maxpool22(self.dropout(F.relu(self.conv31(x), inplace=True)))
        c = self.gnet(x)
        d11 = self.deconv1(c)
        d21 = self.deconv2(c)
        c12 = self.dropout(F.relu(self.conv22(d11), inplace=True))
        c22 = self.dropout(F.relu(self.conv32(d21), inplace=True))
        anet_input = torch.concat([c11, c12], axis=1)
        snet_input = torch.concat([c21, c22], axis=1)
        anet_out = self.anet(anet_input)
        snet_out = self.snet(snet_input)
        merge = torch.concat([anet_out, snet_out], axis=1)
        x = self.final_conv(merge)
        return anet_out.clip(min=0, max=1)
