import torch.nn as nn
import torch.utils.model_zoo as model_zoo


import time


import torch.nn.functional as F

import torch.optim as optim
from torch.autograd import Variable

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os
import time
from utee import misc
print = misc.logger.info


model_urls = {
    'cifar10': 'http://ml.cs.tsinghua.edu.cn/~chenxi/pytorch-models/cifar10-d875770b.pth',
    'cifar100': 'http://ml.cs.tsinghua.edu.cn/~chenxi/pytorch-models/cifar100-3a55a987.pth',
}

def get100(batch_size, data_root='/home/weicheng/HW5/data', train=True, val=True, **kwargs):
    data_root = os.path.expanduser(os.path.join(data_root, 'cifar100-data'))
    num_workers = kwargs.setdefault('num_workers', 1)
    kwargs.pop('input_size', None)
    print("Building CIFAR-100 data loader with {} workers".format(num_workers))
    ds = []
    if train:
        train_loader = torch.utils.data.DataLoader(
            datasets.CIFAR100(
                root=data_root, train=True, download=True,
                transform=transforms.Compose([
                    transforms.Pad(4),
                    transforms.RandomCrop(32),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ])),
            batch_size=batch_size, shuffle=True, **kwargs)
        ds.append(train_loader)


    if val:
        test_loader = torch.utils.data.DataLoader(
            datasets.CIFAR100(
                root=data_root, train=False, download=True,
                transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ])),
            batch_size=batch_size, shuffle=False, **kwargs)
        ds.append(test_loader)
    ds = ds[0] if len(ds) == 1 else ds
    return ds




class CIFAR(nn.Module):
    def __init__(self, features, n_channel, num_classes):
        super(CIFAR, self).__init__()
        assert isinstance(features, nn.Sequential), type(features)
        self.features = features
        self.classifier = nn.Sequential(
            nn.Linear(n_channel, num_classes)
        )
        print(self.features)
        print(self.classifier)

    def forward(self, x):


        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for i, v in enumerate(cfg):
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            padding = v[1] if isinstance(v, tuple) else 1
            out_channels = v[0] if isinstance(v, tuple) else v
            conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=padding)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(out_channels, affine=False), nn.ReLU()]
            else:
                layers += [conv2d, nn.ReLU()]
            in_channels = out_channels
    return nn.Sequential(*layers)



def cifar100(n_channel, pretrained=None):
    cfg = [n_channel, n_channel, 'M', 2*n_channel, 2*n_channel, 'M', 4*n_channel, 4*n_channel, 'M', (8*n_channel, 0), 'M']
    layers = make_layers(cfg, batch_norm=True)
    model = CIFAR(layers, n_channel=8*n_channel, num_classes=100)
    if pretrained is not None:
        #m = model_zoo.load_url(model_urls['cifar100'])
        #state_dict = m.state_dict() if isinstance(m, nn.Module) else m
        #assert isinstance(state_dict, (dict, OrderedDict)), type(state_dict)
        model.load_state_dict(torch.load('log/default/latest.pth'))
    return model




def train():

    misc.logger.init('log/default', 'train_log')
    misc.ensure_dir('log/default')
    train_loader, test_loader = get100(batch_size=200, num_workers=1)
    model = cifar100(n_channel=32,pretrained=1)
    #model = torch.nn.DataParallel(model, device_ids=range(1))
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0)
    best_acc, old_file = 0, None
    t_begin = time.time()
    try:
        # ready to go
        for epoch in range(100):#epoch=100
            model.train()
            if epoch in [80,120]:#decreasing_lr
                optimizer.param_groups[0]['lr'] *= 0.1
            for batch_idx, (data, target) in enumerate(train_loader):
                indx_target = target.clone()
                data, target = Variable(data), Variable(target)

                optimizer.zero_grad()
                output = model(data)
                loss = F.cross_entropy(output, target)
                loss.backward()
                optimizer.step()

                if batch_idx % 100 == 0 and batch_idx > 0:
                    pred = output.data.max(1)[1]  # get the index of the max log-probability
                    correct = pred.cpu().eq(indx_target).sum()
                    acc = correct * 1.0 / len(data)
                    print('Train Epoch: {} [{}/{}] Loss: {:.6f} Acc: {:.4f} lr: {:.2e}'.format(
                        epoch, batch_idx * len(data), len(train_loader.dataset),
                        loss.data[0], acc, optimizer.param_groups[0]['lr']))

            elapse_time = time.time() - t_begin
            speed_epoch = elapse_time / (epoch + 1)
            speed_batch = speed_epoch / len(train_loader)
            eta = speed_epoch * 100 - elapse_time #epoch=100
            print("Elapsed {:.2f}s, {:.2f} s/epoch, {:.2f} s/batch, ets {:.2f}s".format(
                elapse_time, speed_epoch, speed_batch, eta))
            misc.model_snapshot(model, os.path.join('log/default', 'latest.pth'))

            if epoch % 5 == 0:
                model.eval()
                test_loss = 0
                correct = 0
                for data, target in test_loader:
                    indx_target = target.clone()
                    data, target = Variable(data, volatile=True), Variable(target)
                    output = model(data)
                    test_loss += F.cross_entropy(output, target).data[0]
                    pred = output.data.max(1)[1]  # get the index of the max log-probability
                    correct += pred.cpu().eq(indx_target).sum()

                test_loss = test_loss / len(test_loader)  # average over number of mini-batch
                acc = 100. * correct / len(test_loader.dataset)
                print('\tTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
                    test_loss, correct, len(test_loader.dataset), acc))
                if acc > best_acc:
                    new_file = os.path.join('log/default', 'best-{}.pth'.format(epoch))
                    misc.model_snapshot(model, new_file, old_file=old_file, verbose=True)
                    best_acc = acc
                    old_file = new_file
    except Exception as e:
        import traceback
        traceback.print_exc()
    finally:
        print("Total Elapse: {:.2f}, Best Result: {:.3f}%".format(time.time() - t_begin, best_acc))



def forward(x):
    model = cifar100(n_channel=100,pretrained=1)
    return model.forward(x)

def view(img):
    print(forward(img))
    print(img)


def cam(idx = 0):
    vedio = cv2.VideoCapture(idx)
    check,frame = vedio.read()

    if frame is not None:
        for i in len(check):
            check[i] = check[i][3:32:32]
            view(check[i])



if __name__ == '__main__':
    #model = cifar10(128, pretrained='log/cifar10/best-135.pth')
    #model2 = cifar100(128)
    train()
    #embed()
    cam()



