from model.rs_net import RSNet, RefinementNet
from data_loader.data_loader import ISTDLoader
from model.loss import CombinationLoss
import torch
import torch.nn as nn
import os
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import cv2

if __name__ == '__main__':
    # Summary writer for Tensorboard
    writer = SummaryWriter()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    ### Number of Epochs
    num_epochs = 2000
    ### Number of Batch Size    
    batch_size = 4
    ### Learning Rate
    learning_rate = 0.001
    model = RSNet()
    train = ISTDLoader(csv_file='train.csv', root_dir=os.getcwd())
    dataloader = DataLoader(train, batch_size=batch_size, shuffle=True)
    print(model)
    model.to(device=device)
    # model.load_state_dict(torch.load('checkpoints/model_weight_rgb.pth'))
    criterion = CombinationLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    n_total_steps = len(dataloader)
    loss_counter = 0
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for i, data in enumerate(dataloader):
            shadow_image = data['shadow_image'].to(device)
            shape_ = shadow_image.shape
            shadow_image = shadow_image.view(shape_[0], shape_[3], shape_[1], shape_[2])
            shadow_mask_image = data['shadow_mask_image'].to(device)
            shadow_mask_image = shadow_mask_image.view(shape_[0], shape_[3], shape_[1], shape_[2])
            shadow_free_image = data['shadow_free_image'].to(device)
            shadow_free_image = shadow_free_image.view(shape_[0], shape_[3], shape_[1], shape_[2])
            output = model(shadow_image)
            loss = criterion(output, shadow_mask_image, shadow_free_image, shadow_image)
            loss_counter+=1
            writer.add_scalar("current_loss", loss.item(), loss_counter)
            epoch_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if (i + 1) % 4==0:
                print (f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{n_total_steps}], Loss: {loss.item():.4f}')
        ### For now we are saving every weight to see if there was overfitting.
        torch.save(model.state_dict(), f'checkpoints/model_weight_{epoch}.pth')
        epoch_loss /= n_total_steps
        writer.add_scalar("Loss", epoch_loss, epoch)
    print("[+] Training Finished!")
    torch.save(model.state_dict(), 'checkpoints/model_weight_rgb.pth')

