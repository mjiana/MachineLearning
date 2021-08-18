# 학습할 손글씨 이미지 다운로드하고 저장하기
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('./samples/MNIST_data/', one_hot=True)

import tensorflow as tf
import random
import matplotlib.pyplot as plt
print("mnist.train.num_examples", mnist.train.num_examples)  # 55,000장
print("mnist.test.num_examples", mnist.test.num_examples)  # 10,000장
# 임의의 손글씨 숫자 그림을 선택하여 출력
r = random.randint(0, mnist.test.num_examples - 1)
plt.imshow(
    mnist.test.images[r:r + 1].reshape(28, 28),
    cmap="Greys",
    interpolation="nearest"
)
plt.show()
