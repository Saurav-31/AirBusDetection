import matplotlib
matplotlib.use('Agg')
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
from train import train, val
warnings.filterwarnings(action='ignore')

conf = parse_cmd_args()

size = conf['imsize']

data_transform = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(size),
                                     transforms.RandomAffine(0, scale=(1.1, 1.4)),
                                     transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

data_transform1 = transforms.Compose([transforms.Resize(size), transforms.RandomAffine(0, scale=(0.8, 1.3)),
                                      transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

data_transform1 = transforms.Compose([transforms.Resize(256), transforms.ToTensor()])
airbus_data = Airbus(filename='train_ship_segmentations.csv', root_dir=conf['data'], train=True, transform=data_transform1)

test_dataset = Airbus(filename='train_ship_segmentations.csv', root_dir=conf['data'], train=False, transform=data_transform1)

data_iter = iter(airbus_data)

sample = next(data_iter)
sample = next(data_iter)

data, masks, labels = sample['image'], sample['masks'], sample['labels']
#print(data.max(), data.min())
#print(masks.max(), masks.min())
#show_sample3(data, masks)
#print(labels)


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

print('Train Images', len(train_indices))
print('Validation Images', len(val_indices))

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
dataset_sizes = {'train': len(train_indices), 'val': len(val_indices)}

from ptsemseg.models.fcn import fcn8s

model = fcn8s(n_classes=1)
vgg16 = models.vgg16(pretrained=True)
model.init_vgg16_params(vgg16)
model = model.to(device)

if conf['num_gpus'] >1:
    model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))

criterion = nn.BCEWithLogitsLoss()

# Observe that all parameters are being optimized
optimizer_ft = optim.SGD(model.parameters(), lr=conf['lr'], momentum=conf['momentum'])

#sample = next(iter(train_loader))
#out = model(sample['image'].float().to(device))
#out.max()  
#out.shape
#torch.nn.Sigmoid()(out[0])
#torch.nn.Sigmoid()(out[0])>0.5
#sample['masks'][0]

#nn.Softmax2d()(out).max()

#pred = nn.Softmax2d()(out)
from train import accuracy, iou
#print(pred.size())
#accuracy(pred, sample['masks'].float().to(device))
#iou(pred[0], sample['masks'][0].float().to(device))

epochs = 6
since = time.time()
for epoch in range(1, epochs + 1):
    train(model, device, train_loader, optimizer_ft, epoch, criterion, conf)
    val(model, device, validation_loader, epoch, dataset_sizes['val'], conf)

    print("Time Taken for epoch%d: %d sec"%(epoch, time.time()-since))
    since = time.time()
    if epoch % 1 == 0:
        torch.save(model.state_dict(), "./models/FCN8_ep%d.net" % epoch)