#利用numpy手写梯度下降
import numpy as np

#超参数
epochs = 10
lr = 1e-2

#数据集
xs = np.array([1,2,3,4],dtype=np.float32)
ys = np.array([2,4,6,8],dtype=np.float32)
#模型
# y = w * x
w = 0
def forward(x):
    return w * x

def gradient(x,y):
    return 2 * x *(w * x - y)

def loss_fn(x,y):
    pred_y = forward(x)
    return (pred_y - y) ** 2

#训练
for epoch in range(epochs):
    for x,y in zip(xs,ys):
        y_pre = forward(x)
        loss = loss_fn(x,y)
        grad = gradient(x,y)
        w = w - lr * grad
        print('epoch:{:d}, loss:{:.5f},w:{:.4f}'.format(epoch,loss,w))

test_x = 10
print(forward(test_x))