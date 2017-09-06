from neural_network import NeuralNetwork
import torch

class AND():
    x_in = 0
    y_in = 0
    net = 0
    def __init__(self):
        self.net = NeuralNetwork([2, 1])
        self.net.getLayer(0)[0] = torch.FloatTensor([-100, 70, 70])
    def __call__(self, x_in,y_in):
        self.y_in = 1 if y_in == True else 0
        self.x_in = 1 if x_in == True else 0
        return self.forward()

    def forward(self):
        res = self.net.forward([self.x_in,self.y_in])
        return True if res[0]>0.5 else False




class OR():
    x_in = 0
    y_in = 0
    net = 0
    def __init__(self):
        self.net = NeuralNetwork([2, 1])
        self.net.inputMatrix[0] = torch.FloatTensor([[-20, 70, 70]])
    def __call__(self, x_in,y_in):
        self.y_in = 1 if y_in == True else 0
        self.x_in = 1 if x_in == True else 0
        return self.forward()


    def forward(self):

        res = self.net.forward([self.x_in,self.y_in])
        return True if res[0] > 0.5 else False


class NOT():
    x_in = 0
    net = 0
    def __init__(self):
        self.net = NeuralNetwork([1, 1])
        self.net.inputMatrix[0] = torch.FloatTensor([[70, -100]])
    def __call__(self, x_in):
        self.x_in = 1 if x_in == True else 0
        return self.forward()

    def forward(self):

        res = self.net.forward([self.x_in])
        return True if res[0] > 0.5 else False

class XOR():
    x_in = 0
    y_in = 0
    not_gate = 0
    and_gate = 0
    or_gate = 0
    def __init__(self):
        self.not_gate = NOT()
        self.or_gate = OR()
        self.and_gate = AND()
    def __call__(self, x_in,y_in):
        self.y_in = 1 if y_in == True else 0
        self.x_in = 1 if x_in == True else 0
        return self.forward()
    def forward(self):

        res = self.or_gate(self.and_gate(self.x_in,self.not_gate(self.y_in)),self.and_gate(self.not_gate(self.x_in),self.y_in))
        # A xor B = (not A and B) or (A and not B)
        return res
