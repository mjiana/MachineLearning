# 텐서보드를 활용하여 머신러닝 진행과정을 그래프로 출력하기
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
X = tf.placeholder(tf.float32, [None, 2], name='x-input')
Y = tf.placeholder(tf.float32, [None, 1], name='y-input')

with tf.name_scope("layer1") as scope:
    W1 = tf.Variable(tf.random_normal([2, 2]), name='weight1')
    b1 = tf.Variable(tf.random_normal([2]), name='bias1')
    layer1 = tf.sigmoid(tf.matmul(X, W1) + b1)

    w1_hist = tf.summary.histogram("weights1", W1)
    b1_hist = tf.summary.histogram("biases1", b1)
    layer1_hist = tf.summary.histogram("layer1", layer1)

with tf.name_scope("layer2") as scope:
    W2 = tf.Variable(tf.random_normal([2, 1]), name='weight2')
    b2 = tf.Variable(tf.random_normal([1]), name='bias2')
    hypothesis = tf.sigmoid(tf.matmul(layer1, W2) + b2)

    w2_hist = tf.summary.histogram("weights2", W2)
    b2_hist = tf.summary.histogram("biases2", b2)
    hypothesis_hist = tf.summary.histogram("hypothesis", hypothesis)

with tf.name_scope("cost") as scope:
    cost = -tf.reduce_mean(Y * tf.log(hypothesis) + (1-Y) * tf.log(1-hypothesis))
    cost_summ = tf.summary.scalar("cost", cost)

with tf.name_scope("train") as scope:
    train = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, Y), dtype=tf.float32))
accuracy_summ = tf.summary.scalar("accuracy", accuracy)

with tf.Session() as sess:
    merged_summary = tf.summary.merge_all()
    writer = tf.summary.FileWriter("./logs/xor_logs_r0_01")
    writer.add_graph(sess.graph)

    sess.run(tf.global_variables_initializer())

    for step in range(10001):
        summary, _ = sess.run([merged_summary, train],
                              feed_dict={X: x_data, Y: y_data})
        writer.add_summary(summary, global_step=step)

        if step % 100 == 0:
            print(step,
                  sess.run(cost, feed_dict={X: x_data, Y: y_data}),
                  sess.run([W1, W2]))

    h, c, a = sess.run([hypothesis, cost, accuracy],
                       feed_dict={X: x_data, Y: y_data})
    print("\nhypothesis:", h, "\ncost:", c, "\naccuracy:", a)

'''
hypothesis: [[1.9362387e-06]
 [9.9999821e-01]
 [9.9999845e-01]
 [1.9069422e-06]] 
cost: 1.7881409e-06 
accuracy: 1.0
'''
"""
tensorboard 실행방법
1. 현재 파일을 실행하여 로그파일이 생성 후
2. 아나콘다 프롬포트를 실행하고 tensorflow 가상공간 활성화
(base) >activate tensorflow
3. 텐서보드 명령어 입력
(tensorflow) >tensorboard --logdir=C:\\Users\\poimk\\PycharmProjects\\tensorflow36\\logs
* 명령창에 넣을 때는 \\를 \로 바꿔야한다.
4. 맨 마지막 줄에 링크 http://DESKTOP-IV95RFG:6006 으로 인터넷 브라우저 접속
  * 교안처럼 localhost:6006은 접속 불가
"""

