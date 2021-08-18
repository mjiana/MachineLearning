# 학습된 결과를 불러와서 사용하기
import random
import tensorflow as tf
from matplotlib import pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
tf.set_random_seed(777)
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

# 학습하는 부분만 생략 가능
''' 
training_epochs = 15
batch_size = 100
'''

save_file = "./model.ckpt"  # 학습 결과를 불러올 파일명
saver = tf.train.Saver()  # 저장객체 생성

# 저장된 파일을 세션에 다시 불러와서
# 학습과 반복 부분을 생략할 수 있다.
sess = tf.Session()
saver.restore(sess, save_file)

print("Accuracy:", accuracy.eval(session=sess,
                                 feed_dict={X: mnist.test.images,
                                            Y: mnist.test.labels}))
r = random.randint(0, mnist.test.num_examples-1)
print("Label:", sess.run(tf.argmax(mnist.test.labels[r:r+1], 1)))
print("Prediction:", sess.run(tf.argmax(hypothesis, 1),
                              feed_dict={X: mnist.test.images[r:r+1]}))
plt.imshow(
    mnist.test.images[r:r + 1].reshape(28, 28),
    cmap="Greys",
    interpolation="nearest"
)
plt.show()


import numpy as np
print("mnist.test.images[r:r+1]의 차원", np.shape(mnist.test.images[r:r+1]))
# mnist.test.images[r:r+1]의 차원 (1, 784)  # 1행 784열
