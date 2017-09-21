import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as Data
import torchvision



#torch.manual_seed(1)


validaterate = []
trainrate = []


class NnImg2Num(nn.Module):
    train_loader = 0
    def __init__(self):
        super(NnImg2Num,self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(1,16,5,1,2),nn.Sigmoid(),nn.MaxPool2d(2),)
        self.conv2 = nn.Sequential(nn.Conv2d(16,32,5,1,2),nn.Sigmoid(),nn.MaxPool2d(2),)
        self.out = nn.Linear(32*7*7,10)

        train_data = torchvision.datasets.MNIST(
            root='./dataset/',
            train=True,  # this is training data
            transform=torchvision.transforms.ToTensor(),
            download=True,
        )

        self.train_loader = Data.DataLoader(dataset=train_data, batch_size=50, shuffle=True)


    def forward(self,x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0),-1)
        out = self.out(x)
        return out

    def train(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=0.05)  # optimize all cnn parameters
        loss_func = nn.CrossEntropyLoss()  # the target label is not one-hotted
        for epoch in range(1):#EPOCH = 1

            for step,(x,y) in enumerate(self.train_loader):
                b_x = Variable(x)
                b_y = Variable(y)
                output = self.forward(b_x)
                loss = loss_func(output,b_y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            print("epoch = ", epoch)
            base = 0
            point = 0
            for step,(x,y) in enumerate(self.train_loader):
                base += 50
                b_x = Variable(x)
                b_y = Variable(y)
                output = self.forward(b_x)
                pred_y = torch.max(output, 1)[1].data.tolist()
                truey = b_y.data.tolist()
                for i in range(50):
                    point += (1 if pred_y[i] == truey[i] else 0)
            trainrate.append(point/base)
            print("train rate = ",point/base)
            point = 0
            base = 1000
            test_data = torchvision.datasets.MNIST(root='./dataset/', train=False)
            test_x = Variable(torch.unsqueeze(test_data.test_data, dim=1), volatile=True).type(torch.FloatTensor)[
                     :2000] / 255.  # shape from (2000, 28, 28) to (2000, 1, 28, 28), value in range(0,1)
            test_y = test_data.test_labels[:2000]
            test_output = cnn.forward(test_x[:1000])
            pred_y = torch.max(test_output, 1)[1].data.tolist()
            truey = test_y.tolist()
            for i in range(1000):
                point += (1 if pred_y[i] == truey[i] else 0)
            # print(test_output)
            validaterate.append(point/base)
            print("validate rate = ",point/base)
            print("------------------------")

        print(trainrate)
        print(validaterate)










cnn = NnImg2Num()
cnn.train()
'''
test_data = torchvision.datasets.MNIST(root='./dataset/', train=False)
test_x = Variable(torch.unsqueeze(test_data.test_data, dim=1), volatile=True).type(torch.FloatTensor)[:2000]/255.   # shape from (2000, 28, 28) to (2000, 1, 28, 28), value in range(0,1)
test_y = test_data.test_labels[1000:2000]
test_output = cnn.forward(test_x[:10])
#print(test_output)
pred_y = torch.max(test_output, 1)[1].data.numpy().squeeze()
print(pred_y, 'prediction number')
print(test_y[:10].numpy(), 'real number')
'''

point = 0
base = 1000
test_data = torchvision.datasets.MNIST(root='./dataset/', train=False)
test_x = Variable(torch.unsqueeze(test_data.test_data, dim=1), volatile=True).type(torch.FloatTensor)[
         1000:2000] / 255.  # shape from (2000, 28, 28) to (2000, 1, 28, 28), value in range(0,1)
test_y = test_data.test_labels[1000:2000]
test_output = cnn.forward(test_x[:1000])
pred_y = torch.max(test_output, 1)[1].data.tolist()
truey = test_y.tolist()
for i in range(1000):
    point += (1 if pred_y[i] == truey[i] else 0)

print(point/base)

'''
optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)   # optimize all cnn parameters
loss_func = nn.CrossEntropyLoss()   # the target label is not one-hotted

# training and testing
for epoch in range(EPOCH):
    for step, (x, y) in enumerate(train_loader):   # 分配 batch data, normalize x when iterate train_loader
        b_x = Variable(x)   # batch x
        b_y = Variable(y)   # batch y

        output = cnn(b_x)               # cnn output
        loss = loss_func(output, b_y)   # cross entropy loss
        optimizer.zero_grad()           # clear gradients for this training step
        loss.backward()                 # backpropagation, compute gradients
        optimizer.step()                # apply gradients

        if step % 50 == 0:
            test_output = cnn(test_x)
            pred_y = torch.max(test_output, 1)[1].data.squeeze()
            accuracy = sum(pred_y == test_y) / test_y.size(0)
            print(cnn.parameters())
'''
