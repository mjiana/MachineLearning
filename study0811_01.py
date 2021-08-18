# 학습된 데이터가 너무 큰 경우 학습이 안될 수도 있다.
import tensorflow as tf
import random as rd
import numpy as np

# 8 * 5 2차 배열
'''
xy = []
for i in range(40):
    xy.append(rd.uniform(800, 1000))  # 800~1000사이의 실수인 난수가 발생한다.
print(xy)
print(np.reshape(xy, [8, 5]))  # 2차 배열
xy = np.reshape(xy, [8, 5])

출력된 결과를 수정해서 사용
[[845.66782455, 898.77975329, 977147, 871.02708749, 859.83545013],
 [904.85633971, 824.9482724,  868236, 860.62906603, 939.53147435],
 [846.78864649, 927.38789751, 879548, 913.78098949, 896.19225757],
 [814.96402249, 823.79438656, 839640,  917.41400538, 863.31417437],
 [975.87575921, 984.00381222, 825403, 808.98940587, 901.16799293],
 [894.21738523, 973.72685959, 888573,878.91079206, 975.68596961],
 [820.00120331, 870.21864864, 803937, 837.71976429, 828.53722762],
 [874.72017385, 870.62519977, 815900, 894.46519227, 872.74816513]]
'''

xy = np.array([[845.66782455, 898.77975329, 977147, 871.02708749, 859.83545013],
               [904.85633971, 824.9482724, 868236, 860.62906603, 939.53147435],
               [846.78864649, 927.38789751, 879548, 913.78098949, 896.19225757],
               [814.96402249, 823.79438656, 839640, 917.41400538, 863.31417437],
               [975.87575921, 984.00381222, 825403, 808.98940587, 901.16799293],
               [894.21738523, 973.72685959, 888573, 878.91079206, 975.68596961],
               [820.00120331, 870.21864864, 803937, 837.71976429, 828.53722762],
               [874.72017385, 870.62519977, 815900, 894.46519227, 872.74816513]])
x_data = xy[:, 0:-1]
y_data = xy[:, [-1]]

X = tf.placeholder(tf.float32, shape=[None, 4])
Y = tf.placeholder(tf.float32, shape=[None, 1])

W = tf.Variable(tf.random_normal([4, 1], name='weight'))
b = tf.Variable(tf.random_normal([1], name='bias'))

hypothesis = tf.matmul(X, W) + b
# RMSE
cost = tf.reduce_mean(tf.square(hypothesis - Y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-5)
train = optimizer.minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for step in range(101):
    cost_val, hy_val, _ = \
        sess.run([cost, hypothesis, train], feed_dict={X: x_data, Y: y_data})
    print(step, "\ncost:", cost_val, "\nprediction:\n", hy_val)

# 이 예제는 학습 결과가 nan이 나와야 정상이다.
'''
100 
cost: nan 
prediction:
 [[nan]
 [nan]
 [nan]
 [nan]
 [nan]
 [nan]
 [nan]
 [nan]]
'''

