


import cv2
from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os

import sys


dir1 = sys.argv[1]

model1 = models.load_state_dict(torch.load(os.path.join(dir1,'mytraining.pt')))


def cam(idx = 0):
    vedio = cv2.VideoCapture(idx)
    check,frame = vedio.read()

    if frame is not None:
        for i in len(check):
            check[i] = check[i][3:32:32]
            label = model1(check[i])
            print(label)


