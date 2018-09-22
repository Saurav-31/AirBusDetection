import torch.nn as nn
import torch.nn.functional as f
import torch
import numpy as np
import time
from utils import *

def train(model, device, train_loader, optimizer, epoch, criterion):
    model.train()
    acc = 0.0
    avg_loss = 0.0
    since = time.time()
    for batch_idx, sample in enumerate(train_loader):
        data, target = sample['image'], sample['masks']
        data, target = data.float().to(device), target.float().to(device)
        masks = target[:, 0, :, :] #.long()
        
        optimizer.zero_grad()
        output = model(data)
        # print(target[:,0,:,:].size())
        # print(masks.shape)
        # print(output.shape)
        out_masks = nn.Sigmoid()(output)
        # out_masks = nn.Softmax()(output)
        acc += accuracy(nn.Softmax()(output), target)

        loss = criterion(torch.squeeze(output,1), masks)
        loss.backward()
        avg_loss += loss.item()
        optimizer.step()

        # plt.imshow((output[0]*255.0).repeat(3, 1, 1).cpu().detach().numpy().transpose((1, 2, 0)).astype(np.uint8))
        # plt.show()
        if batch_idx > 1500:
            for i in range(4):
                if sample['labels'][i] == 1:
                    show_sample2(torch.squeeze(data[i], 0), out_masks[i].repeat(3, 1, 1), target[i], "img_%d_%d"%(batch_idx, i))

        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{:.0f} ]\tLoss: {:.6f}\tAcc: {:.3f}\tTime(in sec): {:.0f}'.format(
                  epoch, batch_idx * len(data), len(train_loader.dataset) * 0.9,
                  avg_loss / 10, acc/40, time.time() - since))
            # print('Acc: {:.3f}'.format(acc/40))
            avg_loss = 0.0
            acc = 0.0
            since = time.time()



# from sklearn.metrics import confusion_matrix
import torch.nn.functional as f1


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    batch_loss = 0
    with torch.no_grad():
        for i, sample in enumerate(test_loader):
            data, target = sample['image'], sample['masks']
            data, target = data.float().to(device), target.float().to(device)
            masks = target[:,0,:,:]
            output = model(data)
            # print(len(output))
            criterion = nn.BCEWithLogitsLoss()
            loss = criterion(torch.squeeze(output,1), masks)
            test_loss += loss.item()
            batch_loss += loss.item()
            if i %10 == 0:
                print('Loss:{:.0f} '.format(batch_loss/40))
                batch_loss = 0
    test_loss /= len(test_loader)
    print('\nTest set: Average loss: {:.4f})\n'.format(test_loss))


def accuracy(pred, gt):
    return torch.mean((pred == gt).float())


def dice_loss(pred, target):
    smooth = 1.
    # have to use contiguous since they may from a torch.view op
    iflat = pred.contiguous().view(-1)
    tflat = target.contiguous().view(-1)
    intersection = (iflat * tflat).sum()

    A_sum = torch.sum(tflat * iflat)
    B_sum = torch.sum(tflat * tflat)

    return 1 - ((2. * intersection + smooth) / (A_sum + B_sum + smooth))

def iou(pred, target):
    smooth = 1.
    # have to use contiguous since they may from a torch.view op
    iflat = pred.contiguous().view(-1)
    tflat = target.contiguous().view(-1)
    intersection = (iflat * tflat).sum()

    A_sum = torch.sum(tflat * iflat)
    B_sum = torch.sum(tflat * tflat)

    return 1 - ((2. * intersection + smooth) / (A_sum + B_sum + smooth))