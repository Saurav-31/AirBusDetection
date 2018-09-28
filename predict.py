import torch
import matplotlib
matplotlib.use('Agg')
from torchvision import transforms, utils, models
from dataloader import Airbus
from parse_config import *
import numpy as np
import torch.optim as optim
import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import torch.nn as nn
import torchvision.transforms.functional
import warnings
import time
from utils import *
from train import test
warnings.filterwarnings(action='ignore')

conf = parse_cmd_args()

size = conf['imsize']

data_transform1 = transforms.Compose([transforms.Resize(256), transforms.ToTensor()])

test_dataset = Airbus(filename='sample_submission.csv', root_dir=conf['data'], train=False, transform=data_transform1)
print("Test Images: ", len(test_dataset))

data_iter = iter(test_dataset)

sample = next(data_iter)
sample = next(data_iter)

data, masks, labels = sample['image'], sample['masks'], sample['labels']
#print(data.max(), data.min())
#print(masks.max(), masks.min())
#show_sample3(data, masks)
#print(labels)

test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=conf['batch_size'])

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
# torch.cuda.set_device(6)
torch.cuda.set_device(conf['gpu'])
print(device)

dataset_sizes = {'test': len(test_dataset)}

from ptsemseg.models.fcn import fcn8s

model = fcn8s(n_classes=1)
vgg16 = models.vgg16(pretrained=True)
model = model.to(device)
if conf['num_gpus'] >1:
    model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))

model_ft.load_state_dict(torch.load("./models/FCN8_ep4.net"))

from train import accuracy, iou

test(model, device, test_loader, epoch, dataset_sizes['test'], conf)
