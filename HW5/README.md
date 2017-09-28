The homework includes two parts:
1.first part is the compareration of nn package V.S. Lenet-5 package training MNIST dataset. Both of the training missions uses the gpu to accalerate the processes.
    The nn package uses three level. Input(1*28*28) -> Hidden(1*28*28->200) -> Output(200->10)
    The lenet-5 package uses two conv layers and one fully connected layer.
        conv1: input 1,output 16,kernel size 5, stride 1, padding 2
                sigmod()
                maxpooling(2)

        conv2: input 16,output 32,kernel size 5, stride 1, padding 2
                sigmod()
                maxpooling(2)

        Linear: 32*7*7-> 10

    All the training process trains 100 epoches(As in the homework4, I have known that in the epoc 87 the accuracy reaches the highest.)
    NN training uses 1792 seconds
    LENET-5 training uses 1253 seconds.

    The result charts are shown in the directory.


2. Use LENET-5 to train CIFAR-100 dataset.

    The model architecture is below:
    Sequential (
  (0): Conv2d(3, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (1): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=False)
  (2): ReLU ()
  (3): Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (4): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=False)
  (5): ReLU ()
  (6): MaxPool2d (size=(2, 2), stride=(2, 2), dilation=(1, 1))
  (7): Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (8): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=False)
  (9): ReLU ()
  (10): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (11): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=False)
  (12): ReLU ()
  (13): MaxPool2d (size=(2, 2), stride=(2, 2), dilation=(1, 1))
  (14): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (15): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=False)
  (16): ReLU ()
  (17): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (18): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=False)
  (19): ReLU ()
  (20): MaxPool2d (size=(2, 2), stride=(2, 2), dilation=(1, 1))
  (21): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1))
  (22): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=False)
  (23): ReLU ()
  (24): MaxPool2d (size=(2, 2), stride=(2, 2), dilation=(1, 1))
)
Sequential (
  (0): Linear (256 -> 100)
)

AND finally the training accuracy is about 67%


The model also completes the view, cam and forward.