from NeuralNetwork import NeuralNetwork
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as Data
import torchvision


class MyImg2Num():
    net = 0
    def __init__(self):
        self.net = NeuralNetwork([784,10])
    def forward(self,x):
        self.net.forward(x)
    def train(self):
        train_data = torchvision.datasets.MNIST(
            root='./dataset/',
            train=True,  # this is training data
            transform=torchvision.transforms.ToTensor(),
            download=True,
        )

        train_loader = Data.DataLoader(dataset=train_data, batch_size=50, shuffle=True)
        for epoch in range(1):#EPOCH = 1
            for step,(x,y) in enumerate(train_loader):
                b_x = Variable(x)
                b_y = Variable(y)

                #print(b_x.data[0].view([784,1]).eval())
                #print(b_x.data[0].view([1,784]).tolist()[0])
                inputMatrix = []
                targetMatrix = []
                for i in range(50):

                    target = torch.zeros(10)
                    target[b_y.data[i]] = 1
                    #print(b_x.data[0][0].view([1,784]).tolist()[0])
                    inputMatrix.append(b_x.data[i][0].view([1,784]).tolist()[0])
                    targetMatrix.append(target.tolist())
                #print(targetMatrix)
                #print(b_y.data)
                t = b_y.data.tolist()
                for i in range(50):
                    self.net.forward(inputMatrix[i])

                    self.net.backward(targetMatrix[i])
                    self.net.updateParams(0.05)
                    '''
                    out = []
                    out1 = self.net.forward(inputMatrix)
                    for i in range(50):

                        out.append(out1[i].index(max(out1[i])))
                    print(out)
                    print(t)
                    '''



                if step%100 == 0:
                    test_data = torchvision.datasets.MNIST(root='./dataset/', train=False)
                    test_x = Variable(torch.unsqueeze(test_data.test_data, dim=1), volatile=True).type(
                        torch.FloatTensor)[
                             :2000] / 255.  # shape from (2000, 28, 28) to (2000, 1, 28, 28), value in range(0,1)

                    test_y = test_data.test_labels[:2000]
                    rate = 0.0
                    targetlist =[]
                    reslist = []
                    for k in range(20):

                        out = self.net.forward(test_x.data[k].view([1,784]).tolist()[0])
                        reslist.append(out.index(max(out)))
                        targetlist.append(test_y[k])
                        rate += (1 if out.index(max(out)) == test_y[k] else 0)

                    print("step = %d, rate = %f" %(step,rate / 20))
                    print(targetlist)
                    print(reslist)


cnn = MyImg2Num()
cnn.train()