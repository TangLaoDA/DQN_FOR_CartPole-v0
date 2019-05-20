import tensorflow as tf
import gym
import numpy as np
import time
from GomokuGame import Gomoku
import copy

#负责估计值
class QNet:

    def __init__(self,size):

        self.w1 = tf.Variable(tf.truncated_normal([size*size, size*size*6],stddev=0.1),name="q_w1")
        self.b1 = tf.Variable(tf.zeros([size*size*6]),name="q_b1")

        self.w2 = tf.Variable(tf.truncated_normal([size*size*6, size*size*6],stddev=0.1),name="q_w2")
        self.b2 = tf.Variable(tf.zeros(size*size*6),name="q_b2")

        self.w3 = tf.Variable(tf.truncated_normal([size*size*6, size*size], stddev=0.1),name="q_w3")
        self.b3 = tf.Variable(tf.zeros(size*size),name="q_b3")

        self.size=size


    def forward(self,observation):
        #归一化
        observation=observation/2.0
        observation=tf.reshape(observation,[-1,self.size*self.size])
        y = tf.nn.relu(tf.matmul(observation,self.w1)+self.b1)
        y = tf.nn.relu(tf.matmul(y, self.w2) + self.b2)
        y = tf.matmul(y, self.w3) + self.b3

        return y

    def Q_params(self):
        return [self.w1,self.b1,self.w2,self.b2,self.w3,self.b3]
#负责真实值
class TargetQNet:

    def __init__(self,size):
        self.w1 = tf.Variable(tf.truncated_normal([size*size,size*size*6],stddev=0.1),name="target_w1")
        self.b1 = tf.Variable(tf.zeros([size*size*6]),name="target_b1")

        self.w2 = tf.Variable(tf.truncated_normal([size*size*6, size*size*6],stddev=0.1),name="target_w2")
        self.b2 = tf.Variable(tf.zeros(size*size*6),name="target_b2")

        self.w3 = tf.Variable(tf.truncated_normal([size*size*6, size*size], stddev=0.1),name="target_w3")
        self.b3 = tf.Variable(tf.zeros(size*size),name="target_b3")

        self.size=size

    def forward(self,next_observation):
        #归一化
        next_observation=next_observation/2.0
        next_observation = tf.reshape(next_observation, [-1, self.size * self.size])
        y = tf.nn.relu(tf.matmul(next_observation,self.w1)+self.b1)
        y = tf.nn.relu(tf.matmul(y, self.w2) + self.b2)
        y = tf.matmul(y, self.w3) + self.b3

        return y

class Net:

    def __init__(self,size):
        self.observation = tf.placeholder(dtype=tf.float32, shape=[None, size,size])  #nv
        self.action = tf.placeholder(dtype=tf.int32,shape=[None,1])
        self.reward = tf.placeholder(dtype=tf.float32,shape=[None,1])
        self.next_observation = tf.placeholder(dtype=tf.float32, shape=[None, size,size])
        self.done = tf.placeholder(dtype=tf.bool, shape=[None])

        self.qNet = QNet(size)
        self.targetQNet = TargetQNet(size)
        self.size=size

    def forward(self, discount):
        #根据当前状态得到Q值（100个）
        self.pre_qs = self.qNet.forward(self.observation)
        #选择当前动作对应的Q值,神经网络输出2个值，但动作只有一个
        self.pre_q = tf.expand_dims(tf.reduce_sum(tf.multiply(tf.squeeze(tf.one_hot(self.action,self.size*self.size),axis=1),self.pre_qs),axis=1),axis=1)

        #根据下一个状态得到Q(t+1)
        self.next_qs = self.targetQNet.forward(self.next_observation)
        #选择最大的Q值maxQ(t+1)
        self.next_q = tf.expand_dims(tf.reduce_max(self.next_qs,axis=1),axis=1)

        #得到目标Q值。如果是最后一步，只用奖励，否则Q(t)=r(t)+dis*maxQ(t+1)
        self.target_q = tf.where(self.done,self.reward,self.reward + discount * self.next_q)


    def play(self):
        self.qs = self.qNet.forward(self.observation)
        #最大那个Q值的索引就是最大Q值对应的动作
        return tf.argmax(self.qs,axis=1)


    def backward(self):
        #目标Q值-预测Q值
        self.loss = tf.reduce_mean((self.target_q - self.pre_q)**2)
        #RMSPropOptimizer优化器比较平稳
        self.optimizer = tf.train.RMSPropOptimizer(0.01).minimize(self.loss,var_list=self.qNet.Q_params()) #同时优化2个网络？var_list=self.qNet.Q_params()

    def copy_params(self):
        #将Q网络的权重赋予target网络
        return [
            tf.assign(self.targetQNet.w1,self.qNet.w1),
            tf.assign(self.targetQNet.w2, self.qNet.w2),
            tf.assign(self.targetQNet.w3, self.qNet.w3),
            tf.assign(self.targetQNet.b1, self.qNet.b1),
            tf.assign(self.targetQNet.b2, self.qNet.b2),
            tf.assign(self.targetQNet.b3, self.qNet.b3),
        ]

class Game:

    def __init__(self,size):
        # self.env = gym.make('CartPole-v0')
        self.env=Gomoku(size)
        self.size=size

        #用于训练的经验池，先手
        self.experience_pool1 = []
        # 用于训练的经验池，后手
        self.experience_pool2 = []
        #得到游戏初始状态
        self.observation = copy.deepcopy(self.env.reset())

        self.whogo=1

        #创建经验池
        self.CreateExpPool(40000)

        # print("****************************************")
        # print(self.experience_pool1[0][0])
        # print(self.experience_pool1[1][0])
        # print(self.experience_pool1[2][0])
            # print(self.observation)

    #创建经验池
    def CreateExpPool(self,poolsize):
        for i in range(poolsize):
            #一开始随机采样（不知道怎么走）

            if self.whogo == 1:
                # print(self.observation)

                # 先手网络走
                action1 = self.env.sample()
                next_observation, reward, done, info = copy.deepcopy(self.env.step(1,action1))
                tempList=[self.observation[:], reward, action1, next_observation, done]
                self.experience_pool1.append(copy.deepcopy(tempList))  # St,Rt,At,St+1,是否终止
                # print("*********",self.experience_pool1[0][0])
                if done:
                    # 重置游戏
                    self.whogo=1 #游戏重置，还是先手网络走
                    self.observation = copy.deepcopy(self.env.reset())
                else:
                    self.whogo = 2 #游戏未重置，后手网络走
                    self.observation = copy.deepcopy(next_observation)
            elif self.whogo == 2:
                # 后手网络走
                action2 = self.env.sample()
                next_observation, reward, done, info = copy.deepcopy(self.env.step(2,action2))
                tempList = [self.observation[:], reward, action1, next_observation, done]
                self.experience_pool2.append(copy.deepcopy(tempList))  # St,Rt,At,St+1,是否终止
                if done:
                    # 重置游戏
                    self.whogo=1 #游戏重置，还是先手网络走
                    self.observation = copy.deepcopy(self.env.reset())
                else:
                    self.whogo = 1 #游戏未重置，后手网络走完还是先手网络走
                    self.observation = copy.deepcopy(next_observation)
    #获取经验,相当于迷宫游戏的随机状态
    def get_experiences1(self,batch_size):
        experiences = []
        idxs = []
        for _ in range(batch_size):
            #随机取经验（打破相关性）
            idx = np.random.randint(0,len(self.experience_pool1))
            idxs.append(idx)#经验序号
            experiences.append(copy.deepcopy(self.experience_pool1[idx]))#相应的经验
        #idxs是取出经验的序号列表，为了用新的经验替换到老的已训练过的经验
        return idxs,experiences

    def get_experiences2(self,batch_size):
        experiences = []
        idxs = []
        for _ in range(batch_size):
            #随机取经验（打破相关性）
            idx = np.random.randint(0,len(self.experience_pool2))
            idxs.append(idx)#经验序号
            experiences.append(copy.deepcopy(self.experience_pool2[idx]))#相应的经验
        #idxs是取出经验的序号列表，为了用新的经验替换到老的已训练过的经验
        return idxs,experiences

    def reset(self):
        return self.env.reset()
    def chessmanNum(self):
        temp1=[]
        temp2=[]
        len1=len(self.experience_pool1)
        len2=len(self.experience_pool2)
        for i in range(len(self.experience_pool1)):
            observation=self.experience_pool1[i][0]
            temp1.append(self.calChessmanNum(observation))
        for j in range(len(self.experience_pool2)):
            observation=self.experience_pool2[j][0]
            temp2.append(self.calChessmanNum(observation))

        # print(self.experience_pool1[0][0])
        return temp1,len1,temp2,len2
    def calChessmanNum(self,observation):
        num1 = 0  #棋子1的数量
        num2 = 0  # 棋子2的数量
        for i in range(self.size):
            for j in range(self.size):
                if observation[i][j] == 1:
                    num1+=1
                elif observation[i][j] == 2:
                    num2+=1

        return [num1,num2]


    # def render(self):
    #     self.env.render()

size=10
#先手网络
g_first_w1 = tf.Variable(tf.truncated_normal([size*size, size*size*6],stddev=0.1),name="g_first_w1")
g_first_b1 = tf.Variable(tf.zeros([size*size*6]),name="g_first_b1")

g_first_w2 = tf.Variable(tf.truncated_normal([size*size*6, size*size*6],stddev=0.1),name="g_first_w2")
g_first_b2 = tf.Variable(tf.zeros(size*size*6),name="g_first_w2")

g_first_w3 = tf.Variable(tf.truncated_normal([size*size*6, size*size], stddev=0.1),name="g_first_w3")
g_first_b3 = tf.Variable(tf.zeros(size*size),name="g_first_b3")

#后手网络
g_second_w1 = tf.Variable(tf.truncated_normal([size*size, size*size*6],stddev=0.1),name="g_second_w1")
g_second_b1 = tf.Variable(tf.zeros([size*size*6]),name="g_second_b1")

g_second_w2 = tf.Variable(tf.truncated_normal([size*size*6, size*size*6],stddev=0.1),name="g_second_w2")
g_second_b2 = tf.Variable(tf.zeros(size*size*6),name="g_second_b2")

g_second_w3 = tf.Variable(tf.truncated_normal([size*size*6, size*size], stddev=0.1),name="g_second_w3")
g_second_b3 = tf.Variable(tf.zeros(size*size),name="g_second_b3")


if __name__ == '__main__':
    chessSize=10
    game = Game(chessSize)

    # a,len1,b,len2=game.chessmanNum()
    # print(len1,"   ",len2)
    # print(a)
    # print("*****************")
    # print(b)
    # exit()

    #先手网络，下手前黑白棋子等数
    net1 = Net(10)
    net1.forward(0.9)#打折率0.9
    net1.backward()
    copy_op1 = net1.copy_params()
    run_action_op1 = net1.play()#运行游戏

    #后手网络，下手前，敌方棋子多一个
    net2 = Net(10)
    net2.forward(0.9)  # 打折率0.9
    net2.backward()
    copy_op2 = net2.copy_params()
    run_action_op2 = net2.play()  # 运行游戏

    init = tf.global_variables_initializer()

    with tf.Session()  as sess:
        sess.run(init)

        batch_size = 400#一次取200条经验用来训练

        explore1 = 0.1#探索值（前期探索值较大，后期较小）
        explore2 = 0.1  # 探索值（前期探索值较大，后期较小）
        for k in range(10000000):
            #训练网络1==先手网络************************START***************************
            idxs1, experiences1 = game.get_experiences1(batch_size)
            #整理数据（方便输入）
            observations1 = []
            rewards1 = []
            actions1 = []
            next_observations1 = []
            dones1 = []

            for experience in experiences1:
                # print(type(experience[1]))
                # exit()
                observations1.append(experience[0])
                rewards1.append([experience[1]])
                actions1.append([experience[2]])
                next_observations1.append(experience[3])
                dones1.append(experience[4])


            if k % 20 == 0 :
                print("-------------------------------------- copy param -----------------------------------")
                sess.run(copy_op1)
                sess.run(copy_op2)
                # time.sleep(2)


            if k>100000000000:
                while True:
                    # 训练Q网络
                    observations1 = np.array(observations1)
                    next_observations1 = np.array(next_observations1)
                    rewards1 = np.array(rewards1)
                    # print(observations.shape)
                    # exit()
                    _loss, _ = sess.run([net1.loss, net1.optimizer], feed_dict={
                        net1.observation: observations1,
                        net1.action: actions1,
                        net1.reward: rewards1,
                        net1.next_observation: next_observations1,
                        net1.done: dones1
                    })
                    if _loss < 0.1:
                        break
            else:
                # 训练Q网络
                observations1 = np.array(observations1)
                next_observations1 = np.array(next_observations1)
                rewards1 = np.array(rewards1)
                # print(observations.shape)
                # exit()
                _loss, _ = sess.run([net1.loss, net1.optimizer], feed_dict={
                    net1.observation: observations1,
                    net1.action: actions1,
                    net1.reward: rewards1,
                    net1.next_observation: next_observations1,
                    net1.done: dones1
                })

            explore1 -= 0.0001
            if explore1 < 0.0001:
                explore1 = 0.0001

            print("loss1********************************************", _loss, "********************************",explore1)

            count1 = 0
            run_observation1 = copy.deepcopy(game.reset())
            #采集多少经验就要还回去多少经验，刷新经验
            for idx in idxs1:
                # if k > 500:
                #     game.render()#训练500次打印图像查看

                #如果随机值小于探索值，就随机选一个动作作为探索
                if np.random.rand() < explore1:
                    run_action = np.random.randint(0,chessSize*chessSize)
                else:#否则就选Q值最大的那个动作
                    run_action = sess.run(run_action_op1, feed_dict={
                        net1.observation: [run_observation1]
                    })[0]
                    # print(type(run_action))
                    # exit()
                #得到新的经验
                run_next_observation, run_reward, run_done, run_info = copy.deepcopy(game.env.step(1,run_action))
                #覆盖经验池里面的旧的经验，刷新经验
                game.experience_pool1[idx] = copy.deepcopy([run_observation1, run_reward, run_action, run_next_observation,run_done])
                if run_done:
                    run_observation1 = copy.deepcopy(game.reset())
                    count1+=1
                else:
                    #此时1号棋子多一个，应该由后手网络走一步
                    #由后手网络出动作
                    run_observation = copy.deepcopy(run_next_observation)
                    run_action = sess.run(run_action_op2, feed_dict={
                        net2.observation: [run_observation]
                    })[0]
                    run_next_observation, run_reward, run_done, run_info = copy.deepcopy(game.env.step(2, run_action))
                    if run_done:
                        run_observation1 = copy.deepcopy(game.reset())
                    else:
                        run_observation1=copy.deepcopy(run_next_observation)
            print("done1 .......................", count1,"    k   ",k)
            # 训练网络1==先手网络****************END***********************************

            # 训练网络2==后手网络************************START***************************
            idxs2, experiences2 = game.get_experiences2(batch_size)
            # 整理数据（方便输入）
            observations2 = []
            rewards2 = []
            actions2 = []
            next_observations2 = []
            dones2 = []

            for experience in experiences2:
                # print(type(experience[1]))
                # exit()
                observations2.append(experience[0])
                rewards2.append([experience[1]])
                actions2.append([experience[2]])
                next_observations2.append(experience[3])
                dones2.append(experience[4])

            if k % 10 == 0:
                print("-------------------------------------- copy param -----------------------------------")
                sess.run(copy_op2)
                # time.sleep(2)

            if k > 100000:
                while True:
                    # 训练Q网络
                    observations1 = np.array(observations1)
                    next_observations1 = np.array(next_observations1)
                    rewards1 = np.array(rewards1)
                    # print(observations.shape)
                    # exit()
                    _loss, _ = sess.run([net1.loss, net1.optimizer], feed_dict={
                        net1.observation: observations1,
                        net1.action: actions1,
                        net1.reward: rewards1,
                        net1.next_observation: next_observations1,
                        net1.done: dones1
                    })
                    if _loss < 0.1:
                        break
            else:
                # 训练Q网络
                observations2 = np.array(observations2)
                next_observations2 = np.array(next_observations2)
                rewards2 = np.array(rewards2)
                # print(observations.shape)
                # exit()
                _loss, _ = sess.run([net2.loss, net2.optimizer], feed_dict={
                    net2.observation: observations2,
                    net2.action: actions2,
                    net2.reward: rewards2,
                    net2.next_observation: next_observations2,
                    net2.done: dones2
                })

            explore2 -= 0.0001
            if explore2 < 0.0001:
                explore2 = 0.0001

            print("loss2********************************************", _loss, "********************************", explore2)

            count2 = 0
            run_observation2 = copy.deepcopy(game.reset())
            # 采集多少经验就要还回去多少经验，刷新经验
            for idx in idxs2:
                # if k > 500:
                #     game.render()#训练500次打印图像查看

                # 如果随机值小于探索值，就随机选一个动作作为探索
                #此时黑白值相等，必须让1先走一步
                while True:
                    run_action = sess.run(run_action_op1, feed_dict={
                        net1.observation: [run_observation2]
                    })[0]
                    run_next_observation, run_reward, run_done, run_info = copy.deepcopy(game.env.step(1, run_action))
                    if run_done:
                        run_observation2 = copy.deepcopy(game.reset())
                    else:
                        run_observation2 = copy.deepcopy(run_next_observation)
                        break
                if np.random.rand() < explore2:
                    run_action = np.random.randint(0, chessSize * chessSize)
                else:  # 否则就选Q值最大的那个动作
                    run_action = sess.run(run_action_op2, feed_dict={
                        net2.observation: [run_observation2]
                    })[0]
                    # print(type(run_action))
                    # exit()
                # 得到新的经验
                run_next_observation, run_reward, run_done, run_info = copy.deepcopy(game.env.step(2, run_action))
                # 覆盖经验池里面的旧的经验，刷新经验
                game.experience_pool2[idx] = copy.deepcopy(
                    [run_observation2, run_reward, run_action, run_next_observation, run_done])
                if run_done:
                    run_observation2 = copy.deepcopy(game.reset())
                    count2 += 1
                else:
                    # 此时1号棋子多一个，应该由后手网络走一步
                    # 由后手网络出动作
                    run_observation2 = copy.deepcopy(run_next_observation)
                    # run_action = sess.run(run_action_op2, feed_dict={
                    #     net2.observation: [run_observation]
                    # })[0]
                    # run_next_observation, run_reward, run_done, run_info = copy.deepcopy(game.env.step(2, run_action))
                    # if run_done:
                    #     run_observation1 = copy.deepcopy(game.reset())
                    # else:
                    #     run_observation1 = copy.deepcopy(run_next_observation)
            print("done2 .......................", count2, "    k   ", k)
            # 训练网络2==后手网络****************END***********************************

            if k%1000 == 0:
                #先手网络
                sess.run(tf.assign(g_first_w1, net1.qNet.w1))
                sess.run(tf.assign(g_first_b1, net1.qNet.b1))
                sess.run(tf.assign(g_first_w2, net1.qNet.w2))
                sess.run(tf.assign(g_first_b2, net1.qNet.b2))
                sess.run(tf.assign(g_first_w3, net1.qNet.w3))
                sess.run(tf.assign(g_first_b3, net1.qNet.b3))
                #后手网络
                sess.run(tf.assign(g_second_w1, net2.qNet.w1))
                sess.run(tf.assign(g_second_b1, net2.qNet.b1))
                sess.run(tf.assign(g_second_w2, net2.qNet.w2))
                sess.run(tf.assign(g_second_b2, net2.qNet.b2))
                sess.run(tf.assign(g_second_w3, net2.qNet.w3))
                sess.run(tf.assign(g_second_b3, net2.qNet.b3))
                save = tf.train.Saver()
                save_path = save.save(sess, "save_net/save_para{0}.ckpt".format(k))

            if k%200 == 0:
                print("**************************************************")
                print("**************************************************")
                print("**************************************************")
                print("**************************************************")

                # 游戏复位
                run_observation = copy.deepcopy(game.reset())
                while True:
                    #查看棋局
                    game.env.showChessboard()
                    print("先手网络出手***************************************")
                    #先手网络出手
                    run_action = sess.run(run_action_op1, feed_dict={
                        net1.observation: [run_observation]
                    })[0]
                    #先手网络出手后的棋局
                    run_next_observation, run_reward, run_done, run_info = copy.deepcopy(game.env.step(1, run_action))

                    if run_done:
                        print("先手网络出手后游戏挂掉！！！！！！")
                        game.reset()
                        break
                    else:
                        #后手网络出手
                        game.env.showChessboard()
                        print("后手网络出手***************************************")
                        run_action = sess.run(run_action_op2, feed_dict={
                            net2.observation: [run_next_observation]
                        })[0]
                        #后手网络出手后的棋局
                        run_next_observation, run_reward, run_done, run_info = copy.deepcopy(
                            game.env.step(2, run_action))
                        if run_done:
                            print("后手网络出手后游戏挂掉！！！！！！")
                            game.reset()
                            break
                        else:
                            run_observation=copy.deepcopy(run_next_observation)


                print("**************************************************")
                print("**************************************************")
                print("**************************************************")
                print("**************************************************")





