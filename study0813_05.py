# 99% 정확도를 위한 도전
# CNN : Convolutional Neural Network, 합성곱신경망회로
"""
CNN 특징과 용어
상세내용 study0813_05.txt 참조
"""
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
X_img = tf.reshape(X, [-1, 28, 28, 1])  # [개수무관, 가로, 세로, 흑백]
Y = tf.placeholder(tf.float32, [None, 10])

# [3, 3, 1, 32] = 필터크기[가로, 세로, 입력 이미지 개수, 출력 필터개수]
F1 = tf.Variable(tf.random_normal([3, 3, 1, 32], stddev=0.01))
# 가로 한칸 세로 한칸씩 이동
L1 = tf.nn.conv2d(X_img, F1, strides=[1, 1, 1, 1], padding="SAME")
# 결과를 간소화 시킨다.
L1 = tf.nn.relu(L1)
# 2*2 영역에서 가장 큰 값을 선택하고 두칸씩 가로세로 이동,
# 결과는 이미지크기가 절반으로 줄어든다. 28*28 => 14*14
L1 = tf.nn.max_pool(L1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

# 필터크기[가로, 세로, 이미지 개수, 필터개수]
F2 = tf.Variable(tf.random_normal([3, 3, 32, 64], stddev=0.01))
# 가로 한칸 세로 한칸씩 이동
L2 = tf.nn.conv2d(L1, F2, strides=[1, 1, 1, 1], padding="SAME")
# 합성곱 결과 간단
L2 = tf.nn.relu(L2)
# 2*2 영역에서 가장 큰 값을 선택하고 두칸씩 가로세로 이동,
# 결과는 이미지크기가 절반으로 줄어든다. 14*14 => 7*7
# 최대 대표값을 수집한다.
L2 = tf.nn.max_pool(L2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

# fully Connected 에 입력할 값 계산
# 7*7*64= 3136열짜리 이미지로 2D 모양의 최종입력값을 만든다
L2_flat = tf.reshape(L2, [-1, 7*7*64])

# 최종 결과를 얻기 위한 가중치 입력, 출력10개
F3 = tf.get_variable("F3", shape=[7*7*64, 10],
                     initializer=tf.contrib.layers.xavier_initializer())
b = tf.Variable(tf.random_normal([10]))

# 가설의 도출
logits = tf.matmul(L2_flat, F3)+b

cost = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y))
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

correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print("Accuracy:",
      sess.run(accuracy, feed_dict={X: mnist.test.images, Y: mnist.test.labels}))

r = random.randint(0, mnist.test.num_examples-1)
print("Label:", sess.run(tf.argmax(mnist.test.labels[r:r+1], 1)))
print("Prediction:",
      sess.run(tf.argmax(logits, 1), feed_dict={X: mnist.test.images[r:r+1]}))

plt.imshow(mnist.test.images[r:r+1].reshape(28, 28),
           cmap="Greys", interpolation="nearest")
plt.show()

'''
Accuracy: 0.9888
===> study0813_04.py 결과(Accuracy: 0.9832)보다 정확도가 올랐다
Label: [1]
Prediction: [1]
'''
