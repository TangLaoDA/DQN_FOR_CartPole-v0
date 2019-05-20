import numpy as np
from net import QNet
import torch
import torch.nn as nn

q_net=QNet()
target_net=QNet()

optimizer = torch.optim.Adam(q_net.parameters())

#loss_fun
loss_func=nn.L1Loss(size_average=True).cuda()

r=[[-1,-1,-1,-1,0,-1],
   [-1,-1,-1,0,-1,100],
   [-1,-1,-1,0,-1,-1],
   [-1,0,0,-1,0,-1],
   [0,-1,-1,0,-1,100],
   [-1,0,-1,-1,0,100]]

npr=np.array(r)
#归一化
npr=npr/100.0


npq=np.zeros((6,6))


def getAcion(state):
    action = np.random.randint(0, 6)
    # if npr[state][action] > -1:
    #     return action
    # else:
    #     return getAcion(state)
    return action


def qLeaning(state):
    action = getAcion(state) #未作限制
    temp=[[state,action]]
    temp=np.array(temp)
    #归一化
    temp=temp/6
    train_data=torch.Tensor(temp)

    if npr[state][action] > -0.01:
        # print(npr[state][action])

        label=torch.Tensor([npr[state][action]])+torch.Tensor([0.8*readQ(action).max().item()])
    else:
        #犯规，奖励为0
        label=torch.Tensor([[-0.01]])
        # print("********************************")
    # print("label:  ", label)
    actual_value=q_net(train_data)
    # print(actual_value)
    loss=loss_func(actual_value,label)
    print(loss)

    # 清空上一次的梯度
    optimizer.zero_grad()
    # 误差反向传播
    loss.backward()
    # 优化器更新参数
    optimizer.step()
    if action == 5:
        return loss.item()
    else:
        if npr[state][action] > -0.01:

            #没犯规，继续学习
            qLeaning(action)
        else:
            #犯规，游戏挂掉

            return loss.item()


def readQ(state):
    temp=[]
    for i in range(6):
        if npr[state][i]>-1:
            temp.append([state,i])
    train_data = torch.Tensor(temp)
    value=q_net(train_data)

    return value

for i in range(100000):
    init_state = np.random.randint(0, 6)
    loss=qLeaning(init_state)

    if i%10 == 0:
        target_net.asignn(q_net)



