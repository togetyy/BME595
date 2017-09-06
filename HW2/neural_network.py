import math
import torch

class NeuralNetwork():
    inputSize = []  #input+hiddenlayer+ outputsize
    inputMatrix = []    #theta size it is a 3D matrix
    def __init__(self,inputSize):
        self.inputSize = []
        self.inputMatrix = []
        self.inputSize = inputSize
        for i in range(len(inputSize)-1):
            self.inputMatrix.append([])
            self.inputMatrix[i]=(torch.randn(inputSize[i+1],inputSize[i]+1) / math.sqrt(inputSize[i]))  #random initialize theta with a 0 mean, 1/sqrt(size)

    def getLayer(self,layer):
        return self.inputMatrix[layer]#get theta layer

    def forward(self,input):
        if not isinstance(input[0],list): #check if 1d input or 2d input: test whether the 1st input is a list
            input = [input]
        return_result = []
        for m in range(len(input)):

            h_old_temp = [1]
            h_old_temp += input[m]
            h_new_temp = [1]

            for i in range (len(self.inputMatrix)):
                h_new_temp = [1]
                for j in range (len(self.inputMatrix[i])):
                    temp = 0
                    for k in range (len(self.inputMatrix[i][j])):
                        temp += (self.inputMatrix[i][j][k]*h_old_temp[k])
                    temp = 1/(1+math.exp(-temp))    #sigmod the result
                    h_new_temp.append(temp)
                h_old_temp = h_new_temp
            del h_new_temp[0]
            return_result.append(h_new_temp)
        return return_result if len(return_result)>1 else return_result[0]


