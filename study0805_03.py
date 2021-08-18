# 선형회귀 linear_regression =====================================
# set_random_seed(1234)
# 모든 연산에 의해 생성된 난수 시퀀스들이 세션간 반복이 가능하게 하기 위해,
# 그래프 수준의 시드를 설정
# 세션이 달라도 동일한 패턴으로 출력된다.

# random_normal(): 임의의 값을 부여하는데 표준분포곡선을 따른다.

import tensorflow as tf
tf.set_random_seed(777)
x_train = [1, 2, 3]
# y_train = [1, 2, 3]  # 최종결과 : 2000 1.20760815e-05 [1.0040361] [-0.00917497]
# y_train = [2, 4, 6]  # 최종결과 : 2000 5.498538e-06 [2.0027235] [-0.00619128]
y_train = [3, 5, 7]  # 최종결과 : 2000 2.4152463e-05 [2.005708] [0.98702455]

W = tf.Variable(tf.random_normal([1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

hypothesis = x_train * W + b
cost = tf.reduce_mean(tf.square(hypothesis-y_train))
# 하방 기울기 최적화 곡선
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train = optimizer.minimize(cost)

# 실행
sess = tf.Session()
# 전역변수 초기화
sess.run(tf.global_variables_initializer())

for step in range(2001):
    sess.run(train)
    if step % 20 == 0:
        print(step, sess.run(cost), sess.run(W), sess.run(b))

"""
문제) 입력 x [1,2,3] y [2,4,6]일 때, y=2*x+0이론이 나온다.
학습하면 가중치 W는 2에 가까워지는지 조사하시오
=> 2000 5.498538e-06 [2.0027235] [-0.00619128]

문제2) 입력 x [1,2,3] y [3,5,7]일 때, y=2*x+1이론이 나온다.
학습하면 가중치 W는 2에 가까워지고 절편이 1에 가까워 지는지 조사하시오
=> 2000 2.4152463e-05 [2.005708] [0.98702455]
"""
