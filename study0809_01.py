
import tensorflow as tf
tf.set_random_seed(777)

x_data = [1, 2, 3]
y_data = [1, 2, 3]

# random_normal : 정규분포에 따른 난수발생
# random_uniform : 균등분포에 따른 난수발생
W = tf.Variable(tf.random_normal([1]), name='weight')
x = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)
# 절편(b)가 없다.

# 이론, 예측치, 가설
hypothesis = x * W
# 평균 비용 = 평균(제곱(오차)) : 오차를 제곱한 값의 평균
cost = tf.reduce_mean(tf.square(hypothesis - y))

# 최소비용 구하기 == 기울기가 0이 되는 지점 구하기
learning_rate = 0.1
# 경사도 : 미분으로 계산
gradient = tf.reduce_mean((W*x-y)*x)  # 기울기오차 * x <= 2X <= x^2
descent = W - learning_rate * gradient  # 새로운 가중치 구하기
update = W.assign(descent)  # 새로운 가중치 적용

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for step in range(21):
    sess.run(update, feed_dict={x: x_data, y: y_data})
    print(step, sess.run(cost, feed_dict={x: x_data, y: y_data}), sess.run(W))

'''
0 1.9391857 [1.6446238]
....(중략)
20 2.3405278e-11 [1.0000023] <=== cost는 0, W는 1에 가까워 진다.
'''

