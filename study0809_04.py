# 입력데이터가 파일인 경우
import tensorflow as tf
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.set_random_seed(777)

# 데이터가 많을 경우 np를 이용하여 파일 읽음
xy = np.loadtxt("./data/data-01-test-score.csv", delimiter=",", dtype=np.float32)
# 데이터 슬라이싱 => 시작:끝(마지막 값:-1)
x_data = xy[:, 0:-1]  # 각 로우에서 0, 1열 선택
y_data = xy[:, [-1]]  # 각 로우에서 마지막 열 선택

# print(x_data.shape, x_data, len(x_data))
# print(y_data.shape, y_data)

# 키 설정
X = tf.placeholder(tf.float32, shape=[None, 3])
Y = tf.placeholder(tf.float32, shape=[None, 1])

W = tf.Variable(tf.random_normal([3, 1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

# 가설
hypothesis = tf.matmul(X, W) + b
# 비용
cost = tf.reduce_mean(tf.square(hypothesis-Y))
# 최소화 지점
# learning_rate = 0.01부터 0.0001부터는 학습이 안되므로 주의
# 1e-5 == 0.00001
optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-5)
train = optimizer.minimize(cost)

# 실행
sess = tf.Session()
sess.run(tf.global_variables_initializer())

# 학습
for step in range(100000):
    cost_val, hy_val, _ = \
        sess.run([cost, hypothesis, train], feed_dict={X: x_data, Y: y_data})
    if step % 10000 == 0:
        print("step:", step)

# 테스트 - 예측이 될까?
print("[99,99,99] 예측값은?:",
      sess.run(hypothesis, feed_dict={X: [[99, 99, 99]]}))
# 결과 : [[199.26904]]
print("[88, 77, 66], [91, 81, 61] 예측값은?:",
      sess.run(hypothesis, feed_dict={X: [[88, 77, 66], [91, 81, 61]]}))
# 결과 : [[146.60333] [144.1684 ]]

