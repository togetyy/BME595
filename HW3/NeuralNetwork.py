import math
import torch

class NeuralNetwork():
    thetaMatrix = []    #theta size it is a 3D matrix
    dE_dTheta = []

    outputMatrix = []
    def __init__(self,inputSize): #input+hiddenlayer+ output size
        self.thetaMatrix = []

        for i in range(len(inputSize)-1):
            self.thetaMatrix.append([])
            self.thetaMatrix[i]=(torch.randn(inputSize[i+1],inputSize[i]+1) / math.sqrt(inputSize[i]))  #random initialize theta with a 0 mean, 1/sqrt(size)

            self.dE_dTheta.append([])
            self.dE_dTheta[i] = torch.zeros(inputSize[i+1],inputSize[i]+1)
    def getLayer(self,layer):
        return self.thetaMatrix[layer]#get theta layer

    def forward(self,input):
        if not isinstance(input[0],list): #check if 1d input or 2d input: test whether the 1st input is a list
            input = [input]
        return_result = []
        for m in range(len(input)):

            h_old_temp = [1]
            h_old_temp += input[m]
            h_new_temp = [1]
            self.outputMatrix.append([])
            for i in range (len(self.thetaMatrix)):
                h_new_temp = [1]
                for j in range (len(self.thetaMatrix[i])):
                    temp = 0
                    for k in range (len(self.thetaMatrix[i][j])):
                        temp += (self.thetaMatrix[i][j][k]*h_old_temp[k])
                    print(temp)
                    temp = 1/(1+math.exp(-temp))    #sigmod the result
                    h_new_temp.append(temp)

                h_old_temp = h_new_temp[:]
                self.outputMatrix[m].append(h_new_temp)
                del h_new_temp[0]


            return_result.append(h_new_temp)

            self.outputMatrix[m].insert(0,input[m])
            #print(self.outputMatrix)
        return return_result if len(return_result)>1 else return_result[0]


    def backward(self,target,loss):
        delta = [] #δ

        if not isinstance(target[0],list):
            target = [target]
        for i in range(len(target)):
            old_delta = list(map(lambda x:(x[0]-x[1])*(1-x[0])*x[0], zip(self.outputMatrix[i][-1],target[i])))
            delta.append([old_delta])
            for j in range(len(self.outputMatrix[i])-2,-1,-1):
                new_delta = []
                for k in range(1,len(self.thetaMatrix[j][:][0])):

                    temp = 0.0
                    for l in range(len(old_delta)):
                        temp += self.thetaMatrix[j][l][k]*old_delta[l]
                    #temp=sum(list(map(lambda y:y[0]*y[1],zip(old_delta,list(self.thetaMatrix[j][:][k])))))
                    new_delta.append(temp)
                delta[i].insert(0,list(map(lambda x: x[0]*(1-x[0])*x[1], zip(list(self.outputMatrix[i][j]),new_delta))))
                old_delta = new_delta


            print(delta)
            print("--------------------------")

            outputMatrix2 = self.outputMatrix[:]
            for j in range(len(outputMatrix2)):
                for k in range(len(outputMatrix2[j])-1):
                    outputMatrix2[j][k].insert(0,1)
            print(self.outputMatrix)
            for j in range(len(self.thetaMatrix)):

                for k in range(len(self.thetaMatrix[j])):

                    for l in range(len(self.thetaMatrix[j][k])):
                        self.dE_dTheta[j][k][l]+=(delta[i][j+1][k]*self.outputMatrix[i][j][l])
                        print( self.dE_dTheta[j][k][l])





        if loss == 'MSE':
            return

        return



    def updateParams(self,eta):

        for i in range(len(self.thetaMatrix)):
            for j in range(len(self.thetaMatrix[i])):
                for k in range(len(self.thetaMatrix[i][j])):
                    self.thetaMatrix[i][j][k] -= eta/len(self.outputMatrix)*self.dE_dTheta[i][j][k]
                    print(self.thetaMatrix[i][j][k])





a = NeuralNetwork([2,2,2])

a.thetaMatrix[0][0][0] = 0.3500000
a.thetaMatrix[0][0][1] = 0.1500000
a.thetaMatrix[0][0][2] = 0.2000000
a.thetaMatrix[0][1][0] = 0.3500000
a.thetaMatrix[0][1][1] = 0.2500000
a.thetaMatrix[0][1][2] = 0.3000000

a.thetaMatrix[1][0][0] = 0.6000000
a.thetaMatrix[1][0][1] = 0.4000000
a.thetaMatrix[1][0][2] = 0.4500000
a.thetaMatrix[1][1][0] = 0.6000000
a.thetaMatrix[1][1][1] = 0.5000000
a.thetaMatrix[1][1][2] = 0.5500000
#print(a.thetaMatrix)
a.forward([0.05,0.10])
#print(a.outputMatrix)
a.backward([0.01,0.99],'MSE')
a.updateParams(0.5)



