# 선형회귀 linear_regression =====================================
import tensorflow as tf

tf.set_random_seed(777)

W = tf.Variable(tf.random_normal([1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

# 값을 미리 설정하지 않고 placeholder를 통해 설정 후 나중에 부여한다.
x = tf.placeholder(tf.float32, shape=[None])
y = tf.placeholder(tf.float32, shape=[None])

hypothesis = x * W + b
cost = tf.reduce_mean(tf.square(hypothesis - y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train = optimizer.minimize(cost)

# 실행
sess = tf.Session()
# 전역변수 초기화
sess.run(tf.global_variables_initializer())

# x에 [1, 2, 3] 입력 시 라벨 y가 [1, 2, 3]처럼 되도록 학습
for step in range(2001):
    cost_val, W_val, b_val, _ = \
        sess.run([cost, W, b, train], feed_dict={x: [1, 2, 3], y: [1, 2, 3]})
    if step % 20 == 0:
        print(step, cost_val, W_val, b_val)
# 결과
# 2000 1.21343355e-05 [1.0040361] [-0.00917497]

# 학습된 결과에 의거하여 아래의 x값을 입력하면 y는 얼마라고 예측하는가?
print(sess.run(hypothesis, feed_dict={x: [23]}))  # [23.083654]
print(sess.run(hypothesis, feed_dict={x: [2.44, 5, 66]}))  # [2.440673 5.0110054 66.2572]
print(sess.run(hypothesis, feed_dict={x: [13.8]}))  # [13.846522]

# 입력 x: [1, 2, 3], 라벨 y: [1.5, 2.5, 3.5] 학습
for step in range(2001):
    cost_val, W_val, b_val, _ = \
        sess.run([cost, W, b, train], feed_dict={x: [1, 2, 3], y: [1.5, 2.5, 3.5]})
    if step % 20 == 0:
        print(step, cost_val, W_val, b_val)

# 학습된 결과에 의거하여 아래의 x값을 입력하면 y는 얼마라고 예측하는가?
print(sess.run(hypothesis, feed_dict={x: [23]}))  # [23.531622]
print(sess.run(hypothesis, feed_dict={x: [2.44, 5, 66]}))  # [2.940255 5.504161 66.59722]
print(sess.run(hypothesis, feed_dict={x: [13.8]}))  # [14.317587]
# 0.5정도 더 나오도록 학습되었다.

