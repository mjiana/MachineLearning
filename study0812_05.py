# 오차를 감소시키기 위한 방안
# 교안 lab-09-5, 1952줄~
# prop 예제는 이거 하나만 실행한다.
""" 용어와 원리만 기억하기
오차의 역전파(back propagation) : 오차를 역으로(이전단계로) 전파한다.
  -- 상세내용은 study0812_05.txt 참조
"""
import tensorflow as tf

tf.set_random_seed(777)

x_data = [[1.], [2.], [3.]]
y_data = [[1.], [2.], [3.]]

X = tf.placeholder(tf.float32, [None, 1])
Y = tf.placeholder(tf.float32, [None, 1])

# truncated_normal() : 너무 크거나 작은 값은 제외한 랜덤값
W = tf.Variable(tf.truncated_normal([1, 1]))
# 임의의 값 5.0 대입
b = tf.Variable(5.)

# 가설
hypothesis = tf.matmul(X, W)+b
# assert : 조건에 맞지 않으면 에러를 발생하는 의미로 사용
assert hypothesis.shape.as_list() == Y.shape.as_list()
diff = (hypothesis - Y)
0
#
d_l1 = diff
d_b = d_l1

# transpose : 축을 서로 변경한다.
d_w = tf.matmul(tf.transpose(X), d_l1)
print(X, W, d_l1, d_w)
'''
Tensor("Placeholder:0", shape=(?, 1), dtype=float32)
<tf.Variable 'Variable:0' shape=(1, 1) dtype=float32_ref>
Tensor("sub:0", shape=(?, 1), dtype=float32)
Tensor("MatMul_1:0", shape=(1, 1), dtype=float32)
'''

learning_rate = 0.1
step = [tf.assign(W, W-learning_rate*d_w),
        tf.assign(b, b-learning_rate*tf.reduce_mean(d_b))]

RMSE = tf.reduce_mean(tf.square(Y-hypothesis))

# Session()과 InteractiveSession()의 유일한 차이점:
# 생성시 자기 자신을 기본 세션으로 설치한다.
sess = tf.InteractiveSession()
init = tf.global_variables_initializer()
sess.run(init)

for i in range(1000):
    print(i, sess.run([step, RMSE], feed_dict={X: x_data, Y: y_data}))
print("\nhypothesis:\n", sess.run(hypothesis, feed_dict={X: x_data}))

'''
998 [[array([[0.99999726]], dtype=float32), 6.2779877e-06], 5.6464464e-12]
999 [[array([[0.9999973]], dtype=float32), 6.1985147e-06], 5.6464464e-12]

hypothesis:
 [[1.0000035]
 [2.0000007]
 [2.999998 ]]
'''
