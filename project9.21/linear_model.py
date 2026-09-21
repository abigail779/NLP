import torch
import torch.nn as nn
from torch.utils.data import Dataset , DataLoader

from scratch import y_pred
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
Ir = 1e-2
epochs = 10

# xs = torch.tensor([1, 2, 3,4],dtype=torch.float32)
# xs = torch.view(4,1)
# ys = torch.tensor([2, 4, 6,8],dtype=torch.float32)
# ys = torch.view(4,1)

class myDataset(Dataset):
    def __init__(self):
        super(myDataset,self).__init__()
    def _getitem__(self,index):
        pass
    def __len__(self):
        pass
    

# 创建模型

class LinearModel(nn.Module):
    def __init__(self):
        super(LinearModel,self).__init__()
        self.fc = nn.Linear(1,1)

    def forward(self, x):
        return self.fc(x)

model = LinearModel()
model.to(device)

#损失函数
criterion = torch.nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(),lr=Ir)

# 训练
for epoch in range(epochs):
    for(x,y) in zip(xs,ys):

        y_pred = model(x)
        loss = criterion(y_pred,y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f'Epoch {epoch}: Loss {loss.item():.4f}')

#测试
test_x = torch.tensor(10,dtype=torch.float32)
test_x = test_x.to(device)
test_y = model(test_x)
test_y = test_y.to(device)
print(test_y)