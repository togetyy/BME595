from NeuralNetwork import NeuralNetwork


class AND():
    x_in = 0
    y_in = 0
    net = 0
    def __init__(self):
        self.net = NeuralNetwork([2, 1])
        #self.net.getLayer(0)[0] = torch.FloatTensor([-100, 70, 70])
    def __call__(self, x_in,y_in):
        self.y_in = 1 if y_in == True else 0
        self.x_in = 1 if x_in == True else 0
        return self.forward()

    def forward(self):
        res = self.net.forward([self.x_in,self.y_in])
        return True if res[0]>0.5 else False

    def train(self):
        for i in range(1000):
            self.net.forward([True,True])
            self.net.backward([True and True],'MSE')
            self.net.updateParams(0.5)
            self.net.forward([True, False])
            self.net.backward([True and False], 'MSE')
            self.net.updateParams(0.5)
            self.net.forward([False, True])
            self.net.backward([False and True], 'MSE')
            self.net.updateParams(0.5)
            self.net.forward([False, False])
            self.net.backward([False and False], 'MSE')
            self.net.updateParams(0.5)




class OR():
    x_in = 0
    y_in = 0
    net = 0
    def __init__(self):
        self.net = NeuralNetwork([2, 1])
        #self.net.inputMatrix[0] = torch.FloatTensor([[-20, 70, 70]])
    def __call__(self, x_in,y_in):
        self.y_in = 1 if y_in == True else 0
        self.x_in = 1 if x_in == True else 0
        return self.forward()


    def forward(self):

        res = self.net.forward([self.x_in,self.y_in])
        return True if res[0] > 0.5 else False
    def train(self):
        for i in range(1000):
            self.net.forward([True,True])
            self.net.backward([True or True],'MSE')
            self.net.updateParams(0.5)
            self.net.forward([True, False])
            self.net.backward([True or False], 'MSE')
            self.net.updateParams(0.5)
            self.net.forward([False, True])
            self.net.backward([False or True], 'MSE')
            self.net.updateParams(0.5)
            self.net.forward([False, False])
            self.net.backward([False or False], 'MSE')
            self.net.updateParams(0.5)


class NOT():
    x_in = 0
    net = 0
    def __init__(self):
        self.net = NeuralNetwork([1, 1])
        #self.net.inputMatrix[0] = torch.FloatTensor([[70, -100]])
    def __call__(self, x_in):
        self.x_in = 1 if x_in == True else 0
        return self.forward()

    def forward(self):

        res = self.net.forward([self.x_in])
        return True if res[0] > 0.5 else False

    def train(self):
        for i in range (1000):
            self.net.forward([True])
            self.net.backward([not True],'MSE')
            self.net.updateParams(0.5)
            self.net.forward([False])
            self.net.backward([not False], 'MSE')
            self.net.updateParams(0.5)


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

    def train(self):
        self.and_gate.train()
        self.or_gate.train()
        self.not_gate.train()


'''
And = AND()
And.train()
print(And(True,True))
print(And(True,False))
print(And(False,True))
print(And(False,False))

Or = OR()
Or.train()
print(Or(True,True))
print(Or(True,False))
print(Or(False,True))
print(Or(False,False))

Not = NOT()
Not.train()
print(Not(True))
print(Not(False))

Xor = XOR()
Xor.train()
print(Xor(False,True))
print(Xor(False,False))
print(Xor(True,False))
print(Xor(True,True))


'''


