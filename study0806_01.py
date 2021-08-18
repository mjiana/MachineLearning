# 가중치와 절편의 처음 시작값이 미리 설정되어 있는 경우 결과는?
# 학습 후 라벨에 맞도록 가중치와 절편값이 전환되는가?

import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# W 가중치, b 바이어스
# W = tf.Variable([.3], tf.float32)
# b = tf.Variable([-.3], tf.float32)
# 크게 차이나는 가중치와 바이어스를 실행하면?
W = tf.Variable([30.], tf.float32)
b = tf.Variable([-30.], tf.float32)

# x 학습데이터, y 라벨값
x = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)

# 선형이론 y = x * W + b
linear_model = x * W + b

# 비용 함수
loss = tf.reduce_sum(tf.square(linear_model - y))

# 최소화 작업
optimizer = tf.train.GradientDescentOptimizer(0.01)

# 학습
train = optimizer.minimize(loss)

# 실행 ===================
# 실제 데이터 세팅
x_train = [1, 2, 3, 4]
y_train = [0, -1, -2, -3]

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)  # 전역변수 초기화

# 반복 학습
for i in range(1000):
    sess.run(train, {x: x_train, y: y_train})

curr_W, curr_b, curr_loss = sess.run([W, b, loss], {x: x_train, y: y_train})
print("W: %s b: %s loss: %s" % (curr_W, curr_b, curr_loss))

# 결과
# W: [-0.9999969] b: [0.9999908] loss: 5.6999738e-11


# 시작 가중치와 바이어스가 상당한 차이(W 30., b -30.)가 보이는 경우 테스트
# 학습후 라벨에 맞도록 가중치와 절편값이 전환되는가?
# 결과  W: [-0.9999258] b: [0.9997818] loss: 3.1828108e-08
# 결론 : 적합한 가중치와 절편값을 찾아낸다.
