from model.model import DeShadowNet
from data_loader.data_loader import ISTDLoader
from model.loss import CombinationLoss
import torch
import torch.nn as nn
import os
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import cv2
import numpy as np

if __name__ == '__main__':
    ## Set device to CUDA if available
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    ### Load test dataset
    test = ISTDLoader('train.csv', os.getcwd())
    testloader = DataLoader(test)
    criterion = CombinationLoss()
    ### Load SRCNN model to GPU
    model = RSNet()
    # model = nn.DataParallel(model, device_ids=[0, 1])
    model = model.to(device)
    ### Load the saved checkpoint
    model.load_state_dict(torch.load('checkpoints/model_weight_580.pth'))
    ### Run model in evaluation mode
    model.eval()
    ### For images in validation set, pass the image to model.
    for i, data in enumerate(testloader):
        if i == 10:
            break
        shadow_image = data['shadow_image'].to(device)
        shape_ = shadow_image.shape
        shadow_image = shadow_image.view(shape_[0], shape_[3], shape_[1], shape_[2])
        shadow_mask_image = data['shadow_mask_image'].to(device)
        shadow_mask_image = shadow_mask_image.view(shape_[0], shape_[3], shape_[1], shape_[2])
        shadow_free_image = data['shadow_free_image'].to(device)
        shadow_free_image = shadow_free_image.view(shape_[0], shape_[3], shape_[1], shape_[2])

        output = model(shadow_image)
        loss = criterion(output, shadow_mask_image, shadow_free_image, shadow_image)
        print("LOSS: ", loss)
        # print(output.shape)
        print(output.min(), output.max())
        # print(hr_image.min(), hr_image.max())
        print(output.min(), output.max())
        output = output.view(output.shape[2], output.shape[3], 3)
        print(output.shape)
        output = np.abs(output.to('cpu').detach().numpy())* 255
        output = output.astype(np.uint8)
        print(output.shape)
        print(shadow_mask_image.shape)
        print(shadow_mask_image.shape[2], shadow_mask_image.shape[3], 3)
        test = shadow_mask_image.contiguous().view(shadow_mask_image.shape[2], shadow_mask_image.shape[3], 3)
        test = np.abs(test.to('cpu').detach().numpy()) * 255
        test = test.astype(np.uint8)
        print(output.shape, test.shape)
        ### Save the images to output folder.
        cv2.imwrite(f'output/pred_{i}.png', output)
        cv2.imwrite(f'output/out_{i}.png', test)