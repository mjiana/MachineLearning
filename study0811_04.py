# 학습한 데이터를 저장해서 활용이 가능한지?
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf

tf.set_random_seed(777)
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('./samples/MNIST_data/', one_hot=True)

nb_classes = 10  # 0 ~ 9
# 28 * 28 = 784
X = tf.placeholder(tf.float32, [None, 784])
Y = tf.placeholder(tf.float32, [None, nb_classes])

W = tf.Variable(tf.random_normal([784, nb_classes]))
b = tf.Variable(tf.random_normal([nb_classes]))

hypothesis = tf.nn.softmax(tf.matmul(X, W) + b)

cost = tf.reduce_mean(-tf.reduce_sum(Y*tf.log(hypothesis), axis=1))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)

is_correct = tf.equal(tf.argmax(hypothesis, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))

training_epochs = 1000
batch_size = 100

save_file = "./model.ckpt"  # 학습결과를 저장할 파일명
saver = tf.train.Saver()  # 저장객체 생성


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for epoch in range(training_epochs):
        avg_cost = 0
        total_batch = int(mnist.train.num_examples / batch_size)

        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            c, _ = sess.run([cost, optimizer], feed_dict={X: batch_xs, Y: batch_ys})
            avg_cost += c/total_batch

        print("Epoch:", "%04d" % (epoch+1), "cost:", "{:.9f}".format(avg_cost))
    print("Learning Finished")

    saver.save(sess, save_file)  # 학습결과 저장
    print("학습 결과 저장 완료")




