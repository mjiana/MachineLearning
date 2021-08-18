# 교안 677줄
# 다중회귀 Multi linear regression
# 선형회귀는 한가지 종류의 값만 입력되고,
# 다중회귀는 여러 종류의 값들이 입력된다.

import tensorflow as tf
tf.set_random_seed(777)

# 3종류의 데이터 입력
x1_data = [73., 93., 89., 96., 73.]
x2_data = [80., 88., 98., 98., 66.]
x3_data = [75., 93., 90., 100., 73.]
# 라벨값
# y_data = [152., 185., 180., 196., 142.]  # 가설과 차이나는 라벨값
y_data = [228., 274., 277., 294., 212.]  # 가설과 근접한 라벨값

x1 = tf.placeholder(tf.float32)
x2 = tf.placeholder(tf.float32)
x3 = tf.placeholder(tf.float32)
# 텐서 변수
Y = tf.placeholder(tf.float32)

# 각 선형에 대한 가중치
w1 = tf.Variable(tf.random_normal([1]), name='weight1')
w2 = tf.Variable(tf.random_normal([1]), name='weight2')
w3 = tf.Variable(tf.random_normal([1]), name='weight3')
# 절편 값
b = tf.Variable(tf.random_normal([1]), name='bias')

# 가설 설정 : 가중치는 1 바이어스는 0
hypothesis = (x1*w1) + (x2*w2) + (x3*w3) + b
# print(hypothesis)

# 비용 함수
cost = tf.reduce_mean(tf.square(hypothesis-Y))

# 기울기 최적화
optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-5)
train = optimizer.minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for step in range(20001):
    cost_val, hy_val, _ = \
        sess.run([cost, hypothesis, train], feed_dict={x1: x1_data, x2: x2_data, x3: x3_data, Y: y_data})
    if step % 1000 == 0:
        print(step, "cost:", cost_val, "\nprediction:", hy_val)

# 결과값이 라벨값(y_data)에 가까워졌는지 확인
"""
1. 라벨값이 근접한 경우
y_data = [228., 274., 277., 294., 212.]
2만번 학습 시 코스트는 0에 가까워 지고 예측치도 라벨에 가까워진다.

20000 cost: 0.086122975 
prediction: [227.87321 274.28217 277.30743 293.51193 212.04688]

2. 라벨값이 차이가 나는 경우
y_data = [152., 185., 180., 196., 142.]
라벨값과 가설이 차이나면 비용이 크게 나온다.

20000 cost: 2.2749887 
prediction: [149.90654 184.17027 182.08931 195.49323 143.29688]
"""


