# 학습할 손글씨 이미지 다운로드하고 저장하기
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('./samples/MNIST_data/', one_hot=True)
