import gym
env = gym.make('CartPole-v0')
from PIL import Image
import cv2

turn_left = False
turn_right = False
left_count=0
right_count=0

def GetAction():
    global turn_left,turn_right,left_count,right_count
    if (turn_left == False) and (False == turn_right):
        turn_left = True
    if turn_left == True:
        if left_count>6:
            turn_left=False
            turn_right=True
            left_count=0

        left_count +=1
        return 1
    if turn_right == True:
        if right_count > 6:
            turn_left = True
            turn_right = False
            right_count = 0

            right_count += 1
        return 0




for i_episode in range(1):
    observation = env.reset()
    for t in range(1000):
        # env.render()
        # print(observation)
        action = env.action_space.sample()
        # print(action)

        # action=GetAction()
        # action=input("请输入：")
        # action=int(action)
        # print(action)
        #
        # cv2.imshow("sss.jpg", observation)
        # cv2.waitKey(1)
        observation, reward, done, info = env.step(action)
        if done:
            print(reward)
            exit()
        # print(type(observation))
        # print(observation.shape)


        # print(type(reward))
        # print(type(done))
        # print(type(info))
        # print(type(action))
        # print(observation)
        # print("**********************************************")
        # print("reward:  ",reward)
        # print("done: ",done)
        # print("info: ",info)
        # print("action: ",action)
        # exit()

        if done:
            env.reset()
            print("Episode finished after {} timesteps".format(t+1))
            # # print(observation)
            # break
env.close()