import gym
# env = gym.make('MountainCar-v0')
env = gym.make('MsPacman-v0')
env.reset()
for _ in range(1000):
    env.render() #输出游戏场景信息
    env.step(env.action_space.sample()) # take a random action
env.close()