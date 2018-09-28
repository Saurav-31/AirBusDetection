import torch.nn as nn
import torch.nn.functional as f
import torch
import numpy as np
import time
from utils import *
from ptsemseg.metrics import runningScore, averageMeter

def train(model, device, train_loader, optimizer, epoch, criterion, conf):
    model.train()
    # running_metrics_val = runningScore(2)
    print_int = 10
    n = print_int* conf['batch_size']*conf['num_gpus']
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
        out_pred = output.data.max(1)[1].to(device)
        
        # print(output.data.max(1)[1].size())
        # print(nn.Softmax()(output))
        # out_masks = nn.Softmax()(output)
        # acc += accuracy(nn.Softmax()(output), target)
        acc += iou(out_masks.repeat(1, 3, 1, 1), target)
        #print(iou(out_pred.repeat(1, 3, 1, 1), target.long()))
        loss = criterion(torch.squeeze(output,1), masks)
        loss.backward()
        avg_loss += loss.item()
        optimizer.step()
        
        # running_metrics_val.update(target.cpu().numpy(), torch.unsqueeze(out_pred, 1).repeat(1, 3, 1, 1).cpu().numpy())
        # print(out_masks.size())
        # print("IOU:", iou(out_masks.repeat(1, 3, 1, 1), target))
        # plt.imshow((output[0]*255.0).repeat(3, 1, 1).cpu().detach().numpy().transpose((1, 2, 0)).astype(np.uint8))
        # plt.show()
        
        #if batch_idx > 15000:
        #    for i in range(4):
        #        if sample['labels'][i] == 1:
        #            show_sample2(torch.squeeze(data[i], 0), out_masks[i].repeat(3, 1, 1), target[i], "img_%d_%d"%(batch_idx, i))

        if batch_idx % print_int == 0:
            print('Train Epoch: {} [{}/{:.0f} ]\tLoss: {:.6f}\tMeanIOU: {:.3f}\tTime(in sec): {:.0f}'.format(
                  epoch, batch_idx * len(data), len(train_loader.dataset) * 0.9,
                  avg_loss / print_int, acc/print_int, time.time() - since))
            # print('Acc: {:.3f}'.format(acc/40))
            avg_loss = 0.0
            acc = 0.0
            since = time.time()
            #score, class_iou = running_metrics_val.get_scores()
            #print(score, class_iou)



# from sklearn.metrics import confusion_matrix
import torch.nn.functional as f1


def val(model, device, test_loader, epoch, data_size, conf):
    model.eval()
    test_loss = 0
    batch_loss = 0
    print_int = 10
    n = print_int* conf['batch_size']*conf['num_gpus']
    acc = 0.0
    global_iou = 0.0
    print("----------------------------------------------------------------")
    with torch.no_grad():
        for batch_idx, sample in enumerate(test_loader):
            data, target = sample['image'], sample['masks']
            data, target = data.float().to(device), target.float().to(device)
            masks = target[:,0,:,:]
            output = model(data)
            out_masks = nn.Sigmoid()(output)
            
            criterion = nn.BCEWithLogitsLoss()
            loss = criterion(torch.squeeze(output,1), masks)
            
            acc += iou(out_masks.repeat(1, 3, 1, 1), target)
            global_iou += iou(out_masks.repeat(1, 3, 1, 1), target)
            test_loss += loss.item()
            batch_loss += loss.item()
            
            if batch_idx %10 == 0:
                print('Validation Epoch: {} [{}/{:.0f} ]\tLoss:{:.4f}\tMeanIOU:{:.3f}'.format(epoch, batch_idx * len(data), data_size,  batch_loss/print_int, acc/print_int))
                batch_loss = 0
                for i in range(4):
                    if sample['labels'][i] == 1:
                        show_sample2(torch.squeeze(data[i], 0), out_masks[i].repeat(3, 1, 1), target[i], "img_%d_%d"%(batch_idx, i))
                acc = 0.0                
                        
    test_loss /= len(test_loader)
    print('\nTest set: Average loss: {:.4f})\t GlobalIOU: {:.2f}\n'.format(test_loss, conf['batch_size']*conf['num_gpus']*global_iou/data_size))
    
def test(model, device, test_loader, epoch, data_size, conf):
    model.eval()
    test_loss = 0
    batch_loss = 0
    print_int = 10
    n = print_int* conf['batch_size']*conf['num_gpus']
    acc = 0.0
    global_iou = 0.0
    print("----------------------------------------------------------------")
    with torch.no_grad():
        for batch_idx, sample in enumerate(test_loader):
            data = sample['image']
            data = data.float().to(device)
            output = model(data)
            out_masks = nn.Sigmoid()(output)
            for i in range(4):
                show_sample3(torch.squeeze(data[i], 0), out_masks[i].repeat(3, 1, 1), "img_%d_%d"%(batch_idx, i))
    
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
    smooth = 1.0
    # have to use contiguous since they may from a torch.view op
    iflat = pred.contiguous().view(-1)
    tflat = target.contiguous().view(-1)
    intersection = (iflat * tflat).sum()

    A_sum = torch.sum(tflat * iflat)
    B_sum = torch.sum(tflat * tflat)

    return ((2. * intersection + smooth) / (A_sum + B_sum + smooth))
    
    
          