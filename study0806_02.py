# 가중치 W 변화에 따른 비용 cost 의 변화를 그래프로 그리기
# 최소비용 그래프 : 기울기가 0이 되는 지점을 찾는다.
# 결과 : 넓은 U자(밥그릇 모양)의 2차 함수 그래프가 된다.
"""
함수그래프를 기준으로 기울기는 미분으로 구할 수 있으며, 그 기울기가 최소가 되는 것을 찾는다.
  * y=x^2 == 미분 ==> 2x
기울기가 양수인경우 가중치는 감소시키고, 기울기가 음수라면 가중치는 증가시킨다.
결국 기울기가 0이 되는 지점을 찾아내는 것이다.
"""

import tensorflow as tf
# 비용곡선을 확인하기 위한 그래프
import matplotlib.pyplot as plt
tf.set_random_seed(777)

# 입력 x 출력 y
x = [1, 2, 3]
y = [1, 2, 3]
# 가중치
W = tf.placeholder(tf.float32)

hypothesis = x * W
# 비용
cost = tf.reduce_mean(tf.square(hypothesis - y))

# 실행
sess = tf.Session()
sess.run(tf.global_variables_initializer())

# 결과 저장
W_history = []
cost_history = []

# 가중치의 범위가 -30부터 50까지 0.1씩 이동
for i in range(-30, 50):
    curr_W = i * 0.1
    curr_cost = sess.run(cost, feed_dict={W: curr_W})
    W_history.append(curr_W)
    cost_history.append(curr_cost)

# 가중치와 비용을 중심으로 그래프 출력
plt.plot(W_history, cost_history)
plt.show()

