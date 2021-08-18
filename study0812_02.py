# XOR 문제와 멀티 퍼셉트론을 이용한 해결방법
# (퍼셉트론+퍼셉트론+퍼셉트론+퍼셉트론) = 딥러닝의 대두
"""
자세한 내용은 study0812_02.txt 참고
"""

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

# 1단계 뉴럴 네트워크(nn)
# ([입력갯수, 출력갯수])
#   => 다음단계 입력값의 갯수가 2개 이므로
#   => 출력갯수는 최소 2개 이상이 되어야 한다.
W1 = tf.Variable(tf.random_normal([2, 2]), name='weight1')
b1 = tf.Variable(tf.random_normal([2]), name='bias1')
layer1 = tf.sigmoid(tf.matmul(X, W1) + b1)

# 2단계 뉴럴 네트워크(nn)
W2 = tf.Variable(tf.random_normal([2, 1]), name='weight2')
b2 = tf.Variable(tf.random_normal([1]), name='bias2')

hypothesis = tf.sigmoid(tf.matmul(layer1, W2)+b2)

cost = -tf.reduce_mean(Y * tf.log(hypothesis) + (1-Y) * tf.log(1-hypothesis))

train = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)

predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, Y), dtype=tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(10001):
        sess.run(train, feed_dict={X: x_data, Y: y_data})
        if step % 100 == 0:
            print(step,
                  sess.run(cost, feed_dict={X: x_data, Y: y_data}),
                  sess.run([W1, W2]))

    h, c, a = sess.run([hypothesis, cost, accuracy],
                       feed_dict={X: x_data, Y: y_data})
    print("\nhypothesis:", h, "\ncost:", c, "\naccuracy:", a)

'''
hypothesis: [[0.01338218]
 [0.98166394]
 [0.98809403]
 [0.01135799]] 
cost: 0.013844791 
accuracy: 1.0
'''
