from NeuralNetwork import NeuralNetwork
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as Data
import torchvision


class MyImg2Num():
    net = 0
    def __init__(self):
        net = NeuralNetwork([784,32,10])





cnn = MyImg2Num()