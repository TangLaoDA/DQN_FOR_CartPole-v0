import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np

import torch.nn.functional as F#激活函数都在这里

r=[[-1,-1,-1,-1,0,-1],
   [-1,-1,-1,0,-1,100],
   [-1,-1,-1,0,-1,-1],
   [-1,0,0,-1,0,-1],
   [0,-1,-1,0,-1,100],
   [-1,0,-1,-1,0,100]]

npr=np.array(r)

label = [[0.,0.,0.,0.,400.,0.],
         [0.,0.,0.,320.,0.,500.],
         [0.,0.,0.,320.,0.,0.],
         [0.,400.,256.,0.,400.,0.],
         [320.,0.,0.,320.,0.,500.],
         [0.,400.,0.,0.,400.,500.]]

#归一化
label=np.array(label)
label=label/500


EPOCH = 200000000000
BATCH_SIZE = 500
LR = 0.001


class CNN(nn.Module):
    def __init__(self):
        super(CNN,self).__init__()
        self.fullc1 = nn.Linear(2, 128)
        self.fullc2 = nn.Linear(128, 128)
        self.out = nn.Linear(128, 1)
        # torch.nn.init.constant(self.fullc1.weight,0.0)
        # torch.nn.init.constant(self.fullc1.bias, 0.0)
        # torch.nn.init.constant(self.fullc2.weight, 0.0)
        # torch.nn.init.constant(self.fullc2.bias, 0.0)
        # torch.nn.init.constant(self.out.weight, 0.0)
        # torch.nn.init.constant(self.out.bias, 0.0)

    def forward(self,x):
        fc1 = F.relu(self.fullc1(x))
        fc2 = F.relu(self.fullc2(fc1))
        output = self.out(fc2)
        # output = torch.argmax(output, 1).float()
        return output

cnn = CNN().cuda()


#opt
optimizer = torch.optim.Adam(cnn.parameters(),lr=LR)

#loss_fun
loss_func=nn.L1Loss(size_average=True).cuda()

def getAcion(state):
    action = np.random.randint(0, 6)

    return action


#train
for epoch in range(EPOCH):
    init_state = np.random.randint(0, 6)
    action=getAcion(init_state)
    train_data=np.array([[init_state,action]])
    train_data=train_data/6
    batch_x = torch.Tensor(train_data).cuda()

    batch_y = torch.Tensor([label[init_state][action]]).cuda()

    output = cnn(batch_x)
    # print(output)
    # continue

    # 计算误差
    loss = loss_func(output, batch_y)

    # 清空上一次的梯度
    optimizer.zero_grad()
    # 误差反向传播
    loss.backward()
    # 优化器更新参数
    optimizer.step()
    if epoch % 100 == 0:
        temp=[]
        for i in range(6):
            row = []
            for j in range(6):
                testx=torch.Tensor(np.array([[i,j]])/6).cuda()
                value=cnn(testx)
                row.append(value[0][0].item())
            temp.append(row)
        temp=np.array(temp)
        temp=temp*500
        print(temp)







