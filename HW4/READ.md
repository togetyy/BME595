
This homework contains two parts:

1. Fisrt part is using Homework3's NeuralNetwork to build a model to train the mnist dataset.
2. Second part is using Pytorch.nn package module to build a model to train the mnist dataset.

In the first part, I built a model contains two layer,(because when the layer adds up to three, the calculation speed dropped down dramatically.)

The model uses SGD optim stratergy and MSE lossing function with a learning rate of 0.005. Because it is a full connected model, the speed is very slow. If the batch_size is 50, after traing for one epoch, the testing accuracy has up to 99.99%, so I did not draw the picture of error V.S. epoch

In the second part, I built a model contains 3 layers (28*28*1, 14*14*16, 7*7*32), and the output is a size of 10. The model uses SGD optim and CE loss function with the learning rate of 0.05. I found that in the epoch 87, the validation rate has up to the top. So I stop there.

The speed of ecah epoch is about 123 seconds

The train rate V.S. epoch chart is in the folder.
