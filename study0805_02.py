#
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('./samples/MNIST_data/', one_hot=True)

# w=2, b=1
# y = wx + b  <=== 공식 외우기
# x=0 -> y=1 / x=1 -> y=3 / x=2 -> y=5 / x=3 -> y=7
# 결과y = 가중치w * 입력값x + 절편b

# 가중치 w의 값에 따라 선형그래프의 오차값이 작은 것을 찾아가는 것
# 선형 회귀 : w값을 조정하여 직선에 가깝게 만들어 나가는 것

'''
x: 학습할 데이터
placeholder: 변수 예약
float32: 4바이트 실수
[None, 784]: 0행 784열: 이미지 사이즈 (28*28)를 한줄로 만든 것
'''
x = tf.placeholder(tf.float32, [None, 784])

'''
w: 가중치
tf.zeros([784, 10]) : 784개의 값을 입력하면 10개의 결과(0~9사이의 값)를 출력
'''
W = tf.Variable(tf.zeros([784, 10]))

'''
b: 절편값
가중치의 출력 개수와 항상 같다.
'''
b = tf.Variable(tf.zeros([10]))

# 가설 공식 : 결과y = 가중치w * 입력값x + 절편b
# softmax(): 더 정확한 결과를 도출하기 위한 함수, matmul(): 행렬곱
# y: 내가 구한 결과
y = tf.nn.softmax(tf.matmul(x, W)+b)

# y_: 라벨, 정답 제시 10개, 0~9의 값
y_ = tf.placeholder(tf.float32, [None, 10])

# 오차의 합중에서 가장 작은 지점 찾기, 비용곡선
cross_entropy = -tf.reduce_sum(y_*tf.log(y))

# 학습 노드
# 비용의 합계가 가장 작은 지점을 찾는 것을 학습
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

# 학습 실행
# 전역변수 준비 및 초기화
init = tf.global_variables_initializer()

# 세션 : 모든 실제 실행은 세션에서 일어난다.
sess = tf.Session()
sess.run(init)  # 초기화

# 1000번의 학습 실행
for i in range(1000):  # 0~999
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

# 예측
# 내가 구한값과 실제 값을 비교해서 같으면 1, 아니면 0이 담긴다.
# tf.equal(결과값, 라벨값)
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
# 예상 값 : 11111111111011111100111111111
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# 실제로 맞는지 테스트 결과 출력
print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))

# 결과 출력되는데 91%, 0.91x정도의 값이 나온다.
# 0.916
