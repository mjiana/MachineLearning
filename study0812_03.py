# wide & deep learning
# 깊어지고 넓어진다.
# 훨씬 더 낮은 비용을 구할 수 있다.

import tensorflow as tf
import numpy as np

tf.set_random_seed(777)
learning_rate = 0.1
# 데이터 값
x_data = [[0, 0],
          [0, 1],
          [1, 0],
          [1, 1]]
y_data = [[0],
          [1],
          [1],
          [0]]
# 실수 전환
x_data = np.array(x_data, dtype=np.float32)
y_data = np.array(y_data, dtype=np.float32)

# 키설정 : 데이터 입력방식
X = tf.placeholder(tf.float32, [None, 2])
Y = tf.placeholder(tf.float32, [None, 1])

# 1단계 - 2개 입력 10개 출력 * 출력값은 다음단계의 입력값이 된다.
W1 = tf.Variable(tf.random_normal([2, 10]), name='weight1')
b1 = tf.Variable(tf.random_normal([10]), name='bias1')
hypothesis1 = tf.sigmoid(tf.matmul(X, W1)+b1)
# 2단계 - 10개 입력 10개 출력
W2 = tf.Variable(tf.random_normal([10, 10]), name='weight2')
b2 = tf.Variable(tf.random_normal([10]), name='bias2')
hypothesis2 = tf.sigmoid(tf.matmul(hypothesis1, W2)+b2)
# 3단계 - 10개 입력 10개 출력
W3 = tf.Variable(tf.random_normal([10, 10]), name='weight3')
b3 = tf.Variable(tf.random_normal([10]), name='bias3')
hypothesis3 = tf.sigmoid(tf.matmul(hypothesis2, W3)+b3)
# 4단계 - 마지막 단계는 1개 출력
W4 = tf.Variable(tf.random_normal([10, 1]), name='weight4')
b4 = tf.Variable(tf.random_normal([1]), name='bias4')
hypothesis4 = tf.sigmoid(tf.matmul(hypothesis3, W4)+b4)

cost = -tf.reduce_mean(Y * tf.log(hypothesis4) + (1-Y) * tf.log(1-hypothesis4))

train = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)

predicted = tf.cast(hypothesis4 > 0.5, dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, Y), dtype=tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(10001):
        sess.run(train, feed_dict={X: x_data, Y: y_data})
        if step % 100 == 0:
            print(step,
                  sess.run(cost, feed_dict={X: x_data, Y: y_data}))

    h, c, a = sess.run([hypothesis4, cost, accuracy],
                       feed_dict={X: x_data, Y: y_data})
    print("\nhypothesis:", h, "\ncost:", c, "\naccuracy:", a)

'''
hypothesis: [[7.8051287e-04]
 [9.9923813e-01]
 [9.9837923e-01]
 [1.5565894e-03]] 
cost: 0.0011807142
accuracy: 1.0
'''

