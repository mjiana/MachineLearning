# mnist 를 더욱 정확하게 판단할 수 있도록 여러기법을 사용해보자
# 교안 lab-10-2, 2341줄~
"""
----- tf.nn.relu() : 더욱 정확도가 높은 값을 구할 수 있다.
x가 0보다 크면 기울기가 1인 직선, 0보다 작으면 함수값이 0이 된다.
이는 0보다 작은 값을에서 뉴런이 죽을 수 있는 단점을 야기한다.
또한 sigmoid, tanh 함수보다 학습이 빠르고,
연산비용이 적고, 구현이 매우 간단하다는 특징이 있다.
----- AdamOptimizer() : 최적화 함수
안정적인 하강을 위해 사용
"""
# softmax(): 멀티 로지스틱, 확률표시 / sigmoid(): 바이너리 로지스틱, 양분되는 결과

import tensorflow as tf
import random
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data

tf.set_random_seed(777)

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

learning_rate = 0.001
training_epochs = 15
batch_size = 100

X = tf.placeholder(tf.float32, [None, 784])
Y = tf.placeholder(tf.float32, [None, 10])

W1 = tf.Variable(tf.random_normal([784, 256]))
b1 = tf.Variable(tf.random_normal([256]))
L1 = tf.nn.relu(tf.matmul(X, W1)+b1)

W2 = tf.Variable(tf.random_normal([256, 256]))
b2 = tf.Variable(tf.random_normal([256]))
L2 = tf.nn.relu(tf.matmul(L1, W2)+b2)

W3 = tf.Variable(tf.random_normal([256, 10]))
b3 = tf.Variable(tf.random_normal([10]))
hypothesis = tf.matmul(L2, W3)+b3

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
        feed_dict = {X: batch_xs, Y: batch_ys}
        c, _ = sess.run([cost, optimizer], feed_dict=feed_dict)
        avg_cost += c/total_batch
    print("Epoch:", "%04d" % (epoch+1), "cost:", "{:.9f}".format(avg_cost))
print("Learning Finished")

correct_prediction = tf.equal(tf.argmax(hypothesis, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print("Accuracy:",
      sess.run(accuracy, feed_dict={X: mnist.test.images, Y: mnist.test.labels}))

r = random.randint(0, mnist.test.num_examples-1)
print("Label:", sess.run(tf.argmax(mnist.test.labels[r:r+1], 1)))
print("Prediction:",
      sess.run(tf.argmax(hypothesis, 1), feed_dict={X: mnist.test.images[r:r+1]}))

plt.imshow(mnist.test.images[r:r+1].reshape(28, 28),
           cmap="Greys", interpolation="nearest")
plt.show()

'''
Accuracy: 0.9425  
===> relu(), AdamOptimizer()를 적용한 결과 
    기존의 study0811_03.py 결과(Accuracy: 0.8951)보다 정확도가 상승했다.
Label: [6]
Prediction: [6]
'''
