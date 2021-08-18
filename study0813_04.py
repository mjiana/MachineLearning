# 드롭아웃
# xavier deep learning
import tensorflow as tf
import random
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data

tf.set_random_seed(777)

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

learning_rate = 0.001
training_epochs = 15
batch_size = 100

# 입출력값 설정
num_in = 784  # X, 첫번째 입력값
num = 512  # 중간 입출력값
num_out = 10  # Y, 최종 출력값

X = tf.placeholder(tf.float32, [None, num_in])
Y = tf.placeholder(tf.float32, [None, num_out])

# drop out, 연습은 0.5~0.7, 테스트 시 반드시 1
keep_prob = tf.placeholder(tf.float32)

# 깊이를 5단계로, 넓이를 512로 확장
W1 = tf.get_variable("W1", shape=[num_in, num],
                     initializer=tf.contrib.layers.xavier_initializer())
b1 = tf.Variable(tf.random_normal([num]))
L1 = tf.nn.relu(tf.matmul(X, W1)+b1)
L1 = tf.nn.dropout(L1, keep_prob=keep_prob)

W2 = tf.get_variable("W2", shape=[num, num],
                     initializer=tf.contrib.layers.xavier_initializer())
b2 = tf.Variable(tf.random_normal([num]))
L2 = tf.nn.relu(tf.matmul(L1, W2)+b2)
L2 = tf.nn.dropout(L2, keep_prob=keep_prob)

W3 = tf.get_variable("W3", shape=[num, num],
                     initializer=tf.contrib.layers.xavier_initializer())
b3 = tf.Variable(tf.random_normal([num]))
L3 = tf.nn.relu(tf.matmul(L2, W3)+b3)
L3 = tf.nn.dropout(L3, keep_prob=keep_prob)

W4 = tf.get_variable("W4", shape=[num, num],
                     initializer=tf.contrib.layers.xavier_initializer())
b4 = tf.Variable(tf.random_normal([num]))
L4 = tf.nn.relu(tf.matmul(L3, W4)+b4)
L4 = tf.nn.dropout(L4, keep_prob=keep_prob)

W5 = tf.get_variable("W5", shape=[num, num_out],
                     initializer=tf.contrib.layers.xavier_initializer())
b5 = tf.Variable(tf.random_normal([num_out]))
hypothesis = tf.matmul(L4, W5)+b5

cost = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(logits=hypothesis, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for epoch in range(training_epochs):
    avg_cost = 0
    total_batch = int(mnist.train.num_examples / batch_size)

    for i in range(total_batch):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        feed_dict = {X: batch_xs, Y: batch_ys, keep_prob: 0.7}
        c, _ = sess.run([cost, optimizer], feed_dict=feed_dict)
        avg_cost += c/total_batch
    print("Epoch:", "%04d" % (epoch+1), "cost:", "{:.9f}".format(avg_cost))
print("Learning Finished")

correct_prediction = tf.equal(tf.argmax(hypothesis, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print("Accuracy:",
      sess.run(accuracy,
               feed_dict={X: mnist.test.images, Y: mnist.test.labels, keep_prob: 1}))

r = random.randint(0, mnist.test.num_examples-1)
print("Label:", sess.run(tf.argmax(mnist.test.labels[r:r+1], 1)))
print("Prediction:",
      sess.run(tf.argmax(hypothesis, 1),
               feed_dict={X: mnist.test.images[r:r+1], keep_prob: 1}),)

plt.imshow(mnist.test.images[r:r+1].reshape(28, 28),
           cmap="Greys", interpolation="nearest")
plt.show()

'''
Accuracy: 0.9832
===> study0813_03.py 결과(Accuracy: 0.9796)보다 정확도가 올랐다
Label: [6]
Prediction: [6]
'''
