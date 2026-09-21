import numpy as np
import torch

#超参数
epochs = 2
lr = 1e-2

#准备数据集
xs = np.array([1,2,3,4],dtype=np.float32)
ys = np.array([2,4,6,8],dtype=np.float32)

#定义模型
#y = w * x;
w = 5.0
def forward(x):
    return w * x
def loss_fn(x,y):
    y_pred = forward(x)
    loss = (y_pred - y) ** 2
    return loss
def gradient(x,y):
    return 2 * x * (x * w - y)

for epoch in range(epochs):
    for x,y in zip(xs,ys):
        y_pred = forward(x)
        grad = gradient(x,y)
        loss = loss_fn(x,y)
        w = w - lr * grad
        print(epoch,x,y,w,loss)