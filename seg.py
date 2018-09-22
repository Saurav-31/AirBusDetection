from torchvision import transforms, utils, models
from dataloader import Airbus
from parse_config import *
import numpy as np
import torch.optim as optim
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import matplotlib.pyplot as plt
import torch.nn as nn
import torchvision.transforms.functional
import warnings
import time
from utils import *
# from train_model import train, test
from train import train, test
warnings.filterwarnings(action='ignore')

conf = parse_cmd_args()

size = conf['imsize']

data_transform = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(size),
                                     transforms.RandomAffine(0, scale=(1.1, 1.4)),
                                     transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

data_transform1 = transforms.Compose([transforms.Resize(size), transforms.RandomAffine(0, scale=(0.8, 1.3)),
                                      transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

data_transform1 = transforms.Compose([transforms.ToTensor()])
airbus_data = Airbus(filename='train_ship_segmentations.csv', root_dir=conf['data'], train=True, transform=data_transform1)

test_dataset = Airbus(filename='train_ship_segmentations.csv', root_dir=conf['data'], train=False, transform=data_transform1)

data_iter = iter(airbus_data)

sample = next(data_iter)
data, masks, labels = sample['image'], sample['masks'], sample['labels']
print(data.max(), data.min())
# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
# ax1.imshow(data.numpy().transpose(1, 2, 0))
# ax2.imshow(masks)
# plt.show()
# show_sample2(data, masks)
print(labels)

validation_split = conf['val_split']
shuffle_dataset = True
random_seed = conf['seed']

# Creating data indices for training and validation splits:
dataset_size = len(airbus_data)
indices = list(range(dataset_size))
split = int(np.floor(validation_split * dataset_size))
if shuffle_dataset:
    np.random.seed(random_seed)
    np.random.shuffle(indices)
train_indices, val_indices = indices[split:], indices[:split]

# Creating PT data samplers and loaders:
train_sampler = SubsetRandomSampler(train_indices)
valid_sampler = SubsetRandomSampler(val_indices)

train_loader = torch.utils.data.DataLoader(airbus_data, batch_size=conf['batch_size'], sampler=train_sampler)
validation_loader = torch.utils.data.DataLoader(airbus_data, batch_size=conf['batch_size'], sampler=valid_sampler)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=conf['batch_size'])

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
# torch.cuda.set_device(6)
torch.cuda.set_device(conf['gpu'])
print(device)

dataloaders = {'train': train_loader, 'val': validation_loader}
dataset_sizes = {'train': 93664, 'val': 10432}

from ptsemseg.models.fcn import fcn8s

model = fcn8s(n_classes=1)
model = model.to(device)

ngpu = conf['num_gpus']

gpu_ids = []
if ngpu > 1:
  for i in range(ngpu):
    gpu_ids.append(i)
model_ft = torch.nn.DataParallel(model, device_ids=gpu_ids)

criterion = nn.CrossEntropyLoss()
criterion = nn.BCEWithLogitsLoss()

# Observe that all parameters are being optimized
optimizer_ft = optim.SGD(model.parameters(), lr=conf['lr'], momentum=conf['momentum'])
#
# #
# #
# # sample = next(iter(train_loader))
# # out = model(sample['image'].float().to(device))
# #
# #
# # out = model(torch.unsqueeze(data, 0).to(device))
# # out.shape
#
#
# # loss = criterion(out, torch.unsqueeze(masks[0], 0).to(device).long())
#
# # loss = criterion(out.repeat(4,1,1,1), torch.unsqueeze(masks[0], 0).repeat(4,1,1,1).to(device))
# # print(loss)
#
#
# # imshow(torch.squeeze(out)[0].cpu().detach().repeat(3, 1, 1))
# # imshow(torch.squeeze(out).cpu().detach())
#
#
# from train import dice_loss
# dice_loss(masks[0].to(device), out)

epochs = 5
since = time.time()
for epoch in range(1, epochs + 1):
    train(model, device, train_loader, optimizer_ft, epoch, criterion)
    test(model, device, validation_loader)

    print("Time Taken for epoch%d: %d sec"%(epoch, time.time()-since))
    since = time.time()
    if epoch % 5 == 0:
        torch.save(model_ft.state_dict(), "./models/FCN8_ep%d.net" % epoch)



# import torch.nn.functional as F
#
# def cross_entropy2d(input, target, weight=None, size_average=True):
#     # input: (n, c, h, w), target: (n, h, w)
#     n, c, h, w = input.size()
#     # log_p: (n, c, h, w)
#
#     log_p = F.log_softmax(input, dim=1)
#     # log_p: (n*h*w, c)
#     log_p = log_p.transpose(1, 2).transpose(2, 3).contiguous()
#     log_p = log_p[target.view(n, h, w, 1).repeat(1, 1, 1, c) >= 0]
#     log_p = log_p.view(-1, c)
#     # target: (n*h*w,)
#     mask = target >= 0
#     target = target[mask]
#     loss = F.nll_loss(log_p, target, weight=weight, size_average=False)
#     if size_average:
#         loss /= mask.data.sum()
#     return loss
#
# from ptsemseg.loss import cross_entropy2d
# cross_entropy2d(out, masks.long())



# x = torch.unsqueeze(data, 0).to(device)
# for layer in model._modules:
#     out_layer = model._modules[layer](x)
#     print(layer, ":", out_layer.shape)
#     x = out_layer.clone()