import torch
import torch.nn as nn
from torch.autograd import Variable

import torch.nn.functional as F#激活函数都在这里

EPOCH = 2
BATCH_SIZE = 500
LR = 0.001



class QNet(nn.Module):
    def __init__(self):
        super(QNet,self).__init__()
        self.fullc1 = nn.Linear(2, 6*6)
        self.fullc2 = nn.Linear(6*6,6*6)
        self.fullc3 = nn.Linear(6 * 6, 1)

    def forward(self,x):
        #归一化处理
        x=x/5.0
        fc1 = F.relu(self.fullc1(x))
        fc2 = F.relu(self.fullc2(fc1))
        output = self.fullc3(fc2)
        # output = torch.argmax(output, 1).float()
        return output
    def asignn(self,net):
        self.fullc1.weight=net.fullc1.weight
        self.fullc2.weight = net.fullc2.weight
        self.fullc3.weight = net.fullc3.weight
        self.fullc1.bias = net.fullc1.bias
        self.fullc2.bias = net.fullc2.bias
        self.fullc3.bias = net.fullc3.bias

cnn = QNet().cuda()


#opt
optimizer = torch.optim.Adam(cnn.parameters(),lr=LR)

#loss_fun
loss_func=nn.L1Loss(size_average=True).cuda()



#train
if __name__ == '__main__':
    for epoch in range(EPOCH):
        for i, (x, y) in enumerate(train_loader):
            batch_x = Variable(x.view(x.size(0), -1)).cuda()
            y = torch.unsqueeze(y, dim=1)
            batch_y = Variable(y)
            batch_y = (torch.zeros(BATCH_SIZE, 10).scatter_(1, batch_y, 1)).float().cuda()

            output = cnn(batch_x)

            # 计算误差
            loss = loss_func(output, batch_y)

            # 清空上一次的梯度
            optimizer.zero_grad()
            # 误差反向传播
            loss.backward()
            # 优化器更新参数
            optimizer.step()
            if i % 100 == 0:
                print(loss.item())




