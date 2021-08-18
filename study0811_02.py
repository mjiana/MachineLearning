# 큰 값을 작게 바꾸는 min max scalar

import tensorflow as tf
import numpy as np

xy = np.array([[845.66782455, 898.77975329, 977147, 871.02708749, 859.83545013],
               [904.85633971, 824.9482724, 868236, 860.62906603, 939.53147435],
               [846.78864649, 927.38789751, 879548, 913.78098949, 896.19225757],
               [814.96402249, 823.79438656, 839640, 917.41400538, 863.31417437],
               [975.87575921, 984.00381222, 825403, 808.98940587, 901.16799293],
               [894.21738523, 973.72685959, 888573, 878.91079206, 975.68596961],
               [820.00120331, 870.21864864, 803937, 837.71976429, 828.53722762],
               [874.72017385, 870.62519977, 815900, 894.46519227, 872.74816513]])


def MinMaxScalar(data):
    up = data-np.min(data, 0)
    down = np.max(data, 0) - np.min(data, 0)
    return up/(down+1e-5)


xy = MinMaxScalar(xy)
print(xy)

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

for step in range(1000):
    cost_val, hy_val, _ = \
        sess.run([cost, hypothesis, train], feed_dict={X: x_data, Y: y_data})
    if step % 100 == 0:
        print(step, "\ncost:", cost_val, "\nprediction:\n", hy_val)

# 데이터 값의 크기를 줄이면 학습효과가 있다.
'''
900 
cost: 3.6597123 
prediction:
 [[-1.1794086 ]
 [-1.7945771 ]
 [-0.7242466 ]
 [-0.49420947]
 [-2.6427116 ]
 [-1.3933363 ]
 [-1.1148536 ]
 [-1.148698  ]]
'''










