import gym
env = gym.make('CartPole-v1') #创建啊游戏环境
# env = gym.make('SpaceInvaders-v0') #ok
# env = gym.make('Breakout-v0') #打砖块
# env = gym.make('Pong-v0') #OK
# env = gym.make('BeamRider-v0') #OK
# env = gym.make('MsPacman-v0') #OK
# env = gym.make('Seaquest-v0') #OK


env.reset()  #复位
for _ in range(1000):
    env.render() #输出游戏场景信息
    env.step(env.action_space.sample()) # take a random action 从动作空间中
env.close()