import numpy as np
import torch
import tensorflow as tf

# a=[1,2,3,4,5]
# # npa=np.array(a)
# # npa.any()
# #
# # print(npa)

# a=[1,2,3,4]
# b=np.array(a)
# c=torch.Tensor(b)
# print(c)
# x=input("请输入:")
# print(type(x))
# print(x)
# a=False
# b=False
# def test():
#     global a,b
#     if (a == False) and (b == False):
#         a = True
#
# test()
# print(a)
# a=1
# while True:
#     a+=1
#     if a>10:
#         break
# print("ssssssssssss")
#
# a=torch.Tensor([1])
# b=torch.Tensor([2])
# print(a)
# a=b
# print(a)
# print(b)

# a=np.array([1])
# with tf.Session() as sess:
#     print(sess.run(tf.one_hot(a,6)))
# print(type(np.random.randint(0, 6)))
# print("ssssssss")
# print(14**2)
# print(np.zeros((5,5),dtype=np.int32))
# print(5//6)

# a=np.array([1,2,3])
# b=a
# b=np.array([2,3,4])
# print(a,b)
# a=np.array([[1,2,3,4],
#             [5,6,7,8],
#             [10,11,12,13]])
#
# print(a[1][0])
# def test():
#     for i in range(4):
#         for j in range(3):
#             print(a[i][j])
#             if a[i][j] == 3:
#                 return True
#
# print(type(1))
# a=[]
# for i in range(50):
#     a.append([i])
#
# print(a[0][0])
# print(a[1][0])

# a=[1,2,3,4]
# b=[]
# b.append(a[0])
# print(b)
# a[0]=100
# print(b)
# b=a[:]
# print(b)

a=np.array([[[1,2],[3,4]],[[5,6],[7,8]]])
a=tf.Variable([[[1,2],[3,4]],[[5,6],[7,8]]])
# a=a.resize([-1,4])
# print(a)
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    print(sess.run(tf.reshape(a,[-1,4])))
