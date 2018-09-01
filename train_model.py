import torch.nn as nn
import torch.nn.functional as f
import torch
import numpy as np
import time


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.conv3 = nn.Conv2d(64, 128, 3)
        self.fc1 = nn.Linear(115200, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 30)

    def forward(self, x):
        x = f.dropout2d(f.max_pool2d(f.relu(self.conv1(x)), 2), 0.5)
        x = f.dropout2d(f.max_pool2d(f.relu(self.conv2(x)), 2), 0.5)
        x = f.dropout2d(f.max_pool2d(f.relu(self.conv3(x)), 2), 0.5)
        x = x.view(-1, self.num_flat_features(x))
        x = f.dropout(f.relu(self.fc1(x)), 0.5)
        x = f.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    @staticmethod
    def num_flat_features(x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


def train(model, device, train_loader, optimizer, epoch, criterion):
    model.train()
    acc = 0.0
    avg_loss = 0.0
    since = time.time()
    for batch_idx, sample in enumerate(train_loader):
        data, target = sample['image'], sample['labels']
        data, target = data.float().to(device), target.float().to(device)
        # labels = torch.max(target, 1)
        labels = target
        optimizer.zero_grad()
        output = model(data)
        # print(output.size())
        # grads = gradients_multiplier(labels, classes)
        # print(grads)
        # def hook_func(module, grad_i, grad_o):
        #     grad_i = (np.array(grad_i[0]) * grads)
        # criterion.register_backward_hook(hook_func)
        loss = criterion(output, labels.long())
        loss.backward()
        avg_loss += loss.item()
        optimizer.step()
        preds = torch.max(f1.softmax(output), 1)[1]
        acc += accuracy(preds, labels.long())

        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{:.0f} ]\tLoss: {:.6f}\tAccuracy: {:.2f}\tTime(in sec): {:.0f}'.format(
                  epoch, batch_idx * len(data), len(train_loader.dataset) * 0.9,
                  avg_loss / 10, acc / 10, time.time() - since))
            acc = 0.0
            avg_loss = 0.0
            since = time.time()


from sklearn.metrics import confusion_matrix
import torch.nn.functional as f1


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    distribution = torch.zeros(2)
    pred_list, gt_list = [], []
    acc = 0.0
    with torch.no_grad():
        for i, sample in enumerate(test_loader):
            data, target = sample['image'], sample['labels']
            data, target = data.float().to(device), target.float().to(device)
            # labels = torch.max(target, 1)[1]
            labels = target
            output = model(data)
            # print(len(output))
            criterion = nn.CrossEntropyLoss()
            loss = criterion(output, labels.long())
            test_loss += loss.item()
            preds = torch.max(f1.softmax(output), 1)[1]
            acc += accuracy(preds, labels.long())
            pred_list.append(preds.cpu().detach().numpy())
            gt_list.append(labels.cpu().detach().numpy())
            print(acc.item() / (i + 1))
            # print(output.size(), labels.size())
            # if i % 10 == 0:
            #     print(confusion_matrix(labels, preds))
    test_loss /= 600
    acc = (acc * 100) / (i + 1)
    print('\nTest set: Average loss: {:.4f}, Accuracy:({:0.3f}%)\n'.format(
        test_loss, acc))
    return pred_list, gt_list


def accuracy(pred, gt):
    return torch.mean((pred == gt).float())


def gradients_multiplier(labels, classes):
    grad_mul = {'horse': 0.085462, 'squirrel': 0.062154, 'antelope': 0.053462, 'german+shepherd': 0.052846,
                'collie': 0.052308, 'seal': 0.051154, 'buffalo': 0.046615, 'grizzly+bear': 0.044846,
                'otter': 0.039923, 'ox': 0.038538, 'persian+cat': 0.037769, 'rhinoceros': 0.036692,
                'chimpanzee': 0.036692, 'moose': 0.036615,
                'hippopotamus': 0.035692, 'bobcat': 0.032154, 'wolf': 0.031000, 'chihuahua': 0.029692,
                'dalmatian': 0.027615, 'raccoon': 0.026615, 'siamese+cat': 0.026231, 'bat': 0.019692,
                'rat': 0.016923, 'killer+whale': 0.014846, 'spider+monkey': 0.014538, 'weasel': 0.014154,
                'walrus': 0.011385, 'beaver': 0.010231, 'mouse': 0.009538, 'mole': 0.004615}
    # print(labels)
    grads = []
    for i in range(len(labels)):
        one_hot = np.zeros(len(classes))
        # print("Label:", labels[i])
        # print(int(labels[i]))
        grad_val = 1 / grad_mul[classes[int(labels[i])]]
        one_hot[int(labels[i].item())] = grad_val
        grads.append(one_hot)
    return grads


def class_weights(classes):
    grad_mul = {'horse': 0.085462, 'squirrel': 0.062154, 'antelope': 0.053462, 'german+shepherd': 0.052846,
                'collie': 0.052308, 'seal': 0.051154, 'buffalo': 0.046615, 'grizzly+bear': 0.044846,
                'otter': 0.039923, 'ox': 0.038538, 'persian+cat': 0.037769, 'rhinoceros': 0.036692,
                'chimpanzee': 0.036692, 'moose': 0.036615,
                'hippopotamus': 0.035692, 'bobcat': 0.032154, 'wolf': 0.031000, 'chihuahua': 0.029692,
                'dalmatian': 0.027615, 'raccoon': 0.026615, 'siamese+cat': 0.026231, 'bat': 0.019692,
                'rat': 0.016923, 'killer+whale': 0.014846, 'spider+monkey': 0.014538, 'weasel': 0.014154,
                'walrus': 0.011385, 'beaver': 0.010231, 'mouse': 0.009538, 'mole': 0.004615}
    # print(labels)
    grads = []
    for i in classes:
        grad_val = 1 / grad_mul[i]
        grads.append(grad_val)
    return grads

