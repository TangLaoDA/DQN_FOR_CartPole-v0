import tensorflow as tf
import gym
import numpy as np
import time

r=[[-1,-1,-1,-1,0,-1],
   [-1,-1,-1,0,-1,100],
   [-1,-1,-1,0,-1,-1],
   [-1,0,0,-1,0,-1],
   [0,-1,-1,0,-1,100],
   [-1,0,-1,-1,0,100]]

npr=np.array(r)
#归一化
npr=npr/100.0

#负责估计值
class QNet:

    def __init__(self):

        self.w1 = tf.Variable(tf.truncated_normal([1, 30],stddev=0.1))
        self.b1 = tf.Variable(tf.zeros([30]))

        self.w2 = tf.Variable(tf.truncated_normal([30, 30],stddev=0.1))
        self.b2 = tf.Variable(tf.zeros(30))

        self.w3 = tf.Variable(tf.truncated_normal([30, 6], stddev=0.1))
        self.b3 = tf.Variable(tf.zeros(6))


    def forward(self,observation):
        #归一化处理
        observation=observation/5.0
        y = tf.nn.relu(tf.matmul(observation,self.w1)+self.b1)
        y = tf.nn.relu(tf.matmul(y, self.w2) + self.b2)
        y = tf.matmul(y, self.w3) + self.b3

        return y

    def Q_params(self):
        return [self.w1,self.b1,self.w2,self.b2,self.w3,self.b3]
#负责真实值
class TargetQNet:

    def __init__(self):
        self.w1 = tf.Variable(tf.truncated_normal([1, 30],stddev=0.1))
        self.b1 = tf.Variable(tf.zeros([30]))

        self.w2 = tf.Variable(tf.truncated_normal([30, 30],stddev=0.1))
        self.b2 = tf.Variable(tf.zeros(30))

        self.w3 = tf.Variable(tf.truncated_normal([30, 6], stddev=0.1))
        self.b3 = tf.Variable(tf.zeros(6))

    def forward(self,next_observation):
        # 归一化处理
        next_observation = next_observation / 5.0
        y = tf.nn.relu(tf.matmul(next_observation,self.w1)+self.b1)
        y = tf.nn.relu(tf.matmul(y, self.w2) + self.b2)
        y = tf.matmul(y, self.w3) + self.b3

        return y

class Net:

    def __init__(self):
        self.observation = tf.placeholder(dtype=tf.float32, shape=[None, 1])  #nv
        self.action = tf.placeholder(dtype=tf.int32,shape=[None,1])
        self.reward = tf.placeholder(dtype=tf.float32,shape=[None,1])
        self.next_observation = tf.placeholder(dtype=tf.float32, shape=[None, 1])
        self.done = tf.placeholder(dtype=tf.bool, shape=[None])

        self.qNet = QNet()
        self.targetQNet = TargetQNet()

    def forward(self, discount):

        #根据当前状态得到Q值（两个）
        self.pre_qs = self.qNet.forward(self.observation)
        #选择当前动作对应的Q值,神经网络输出2个值，但动作只有一个
        self.pre_q = tf.expand_dims(tf.reduce_sum(tf.multiply(tf.squeeze(tf.one_hot(self.action,6),axis=1),self.pre_qs),axis=1),axis=1)

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

def step(state,action):
    state=state[0]
    #state和action必须是整数
    if npr[state][action] == -0.01: #游戏挂掉
        # print("游戏挂掉")
        next_observation=np.random.randint(0, 6)
        reward=-0.01
        done = True
        info={}
    elif npr[state][action] == 0.0:
        # print("游戏正常运行")
        next_observation = action
        reward = 0.0
        done = False
        info = {}
    elif npr[state][action] == 1.0: #游戏通关
        # print("游戏通关")
        next_observation=np.random.randint(0, 6)
        reward = 1.0
        done = False
        info = {}

    return np.array([next_observation],dtype=np.int32),reward,done,info

# step(1,5)
# exit()

class Game:

    def __init__(self):
        # self.env = gym.make('CartPole-v0')

        #用于训练的经验池
        self.experience_pool = []
        #得到游戏初始状态
        # self.observation = self.env.reset()
        self.observation=np.array([np.random.randint(0, 6)])

        #创建经验
        for i in range(10000):
            #一开始随机采样（不知道怎么走）
            # action = self.env.action_space.sample()
            action = np.random.randint(0, 6)
            next_observation, reward, done, info = step(self.observation,action)
            self.experience_pool.append([self.observation, reward, action, next_observation, done])#St,Rt,At,St+1,是否终止
            if done:
                #重置游戏
                self.observation = np.array([np.random.randint(0, 6)],dtype=np.int32)
            else:

                self.observation = next_observation
    #获取经验,相当于迷宫游戏的随机状态
    def get_experiences(self,batch_size):
        experiences = []
        idxs = []
        for _ in range(batch_size):
            #随机取经验（打破相关性）
            idx = np.random.randint(0,len(self.experience_pool))
            idxs.append(idx)#经验序号
            experiences.append(self.experience_pool[idx])#相应的经验
        #idxs是取出经验的序号列表，为了用新的经验替换到老的已训练过的经验
        return idxs,experiences

    def reset(self):
        return self.env.reset()

    def render(self):
        self.env.render()

if __name__ == '__main__':
    game = Game()

    net = Net()
    net.forward(0.9)#打折率0.9
    net.backward()
    copy_op = net.copy_params()
    run_action_op = net.play()#运行游戏

    init = tf.global_variables_initializer()

    with tf.Session()  as sess:
        sess.run(init)

        batch_size = 200#一次取200条经验用来训练

        explore = 0.1#探索值（前期探索值较大，后期较小）
        for k in range(10000000):
            idxs, experiences = game.get_experiences(batch_size)
            #整理数据（方便输入）
            observations = []
            rewards = []
            actions = []
            next_observations = []
            dones = []

            for experience in experiences:
                # print(type(experience[1]))
                # exit()
                observations.append(experience[0])
                rewards.append([experience[1]])
                actions.append([experience[2]])
                next_observations.append(experience[3])
                dones.append(experience[4])


            if k % 10 == 0 :
                print("-------------------------------------- copy param -----------------------------------")
                sess.run(copy_op)
                # time.sleep(2)


            if k>100000:
                while True:
                    # 训练Q网络
                    observations = np.array(observations)
                    next_observations = np.array(next_observations)
                    rewards = np.array(rewards)
                    # print(observations.shape)
                    # exit()
                    _loss, _ = sess.run([net.loss, net.optimizer], feed_dict={
                        net.observation: observations,
                        net.action: actions,
                        net.reward: rewards,
                        net.next_observation: next_observations,
                        net.done: dones
                    })
                    if _loss < 0.1:
                        break
            else:
                # 训练Q网络
                observations = np.array(observations)
                next_observations = np.array(next_observations)
                rewards = np.array(rewards)
                # print(observations.shape)
                # exit()
                _loss, _ = sess.run([net.loss, net.optimizer], feed_dict={
                    net.observation: observations,
                    net.action: actions,
                    net.reward: rewards,
                    net.next_observation: next_observations,
                    net.done: dones
                })



            explore -= 0.0001
            if explore < 0.0001:
                explore = 0.0001

            print("********************************************", _loss, "********************************",explore)

            count = 0
            # run_observation = game.reset()
            run_observation=np.array([np.random.randint(0, 6)],dtype=np.int32)
            #采集多少经验就要还回去多少经验，刷新经验
            for idx in idxs:
                # if k > 500:
                #     game.render()#训练500次打印图像查看

                #如果随机值小于探索值，就随机选一个动作作为探索
                if np.random.rand() < explore:
                    run_action = np.random.randint(0,6)
                else:#否则就选Q值最大的那个动作
                    run_action = sess.run(run_action_op, feed_dict={
                        net.observation: [run_observation]
                    })[0]
                    # print(type(run_action))
                    # exit()
                #得到新的经验
                run_next_observation, run_reward, run_done, run_info = step(run_observation,run_action)
                #覆盖经验池里面的旧的经验，刷新经验
                game.experience_pool[idx] = [run_observation, run_reward, run_action, run_next_observation,run_done]
                if run_done:
                    run_observation = np.array([np.random.randint(0, 6)],dtype=np.int32)
                    count+=1
                else:
                    run_observation = run_next_observation
            print("done .......................", count,"    k   ",k)

            if k>1200:
                print("**************************************************")
                print("**************************************************")
                print("**************************************************")
                print("**************************************************")
                for i in range(1):
                    state = np.array([2], dtype=np.int32)
                    while True:
                        run_action = sess.run(run_action_op, feed_dict={
                            net.observation: [state]
                        })[0]
                        print(run_action)
                        state, run_reward, run_done, run_info = step(state, run_action)
                        if run_reward == 1.0:
                            break





