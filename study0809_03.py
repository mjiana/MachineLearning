# 교안 727줄
# 앞의 예제는 입력값이 3종류로 고정되어 있었다.
# 만약 입력값의 종류가 n개일 때 처리방법은?
# ===> 행렬곱 이용 matmul()
"""
안쪽이 일치해야 행렬곱이 가능하고, 앞의 행과 뒤의 열 크기도 출력된다.
앞행렬 * 뒤행렬 ===> 결과행렬
2행 2열 * 2행 2열 ===> 2행 2열
2행 3열 * 2행 3열 ===> 행렬곱 불가능
2행 3열 * 3행 2열 ===> 3행 3열
3행 2열 * 2행 3열 ===>  3행 3열
"""
import tensorflow as tf
tf.set_random_seed(777)

# 5행 3열 구조
x_data = [[73., 80., 75.],
          [93., 88., 93.],
          [89., 91., 90.],
          [96., 98., 100.],
          [73., 66., 70.]]
# 5행 1열 구조
'''
y_data = [[152.],
          [185.],
          [180.],
          [196.],
          [142.]]
'''
y_data = [[228.],
          [274.],
          [270.],
          [294.],
          [209.]]

# 텐서 키 변수 : 3열짜리 데이터
X = tf.placeholder(tf.float32, shape=[None, 3])
# 예측값(라벨) : 1열짜리 데이터
Y = tf.placeholder(tf.float32, shape=[None, 1])

# X 3개가 들어와서 Y 1개가 나간다.
# 가중치(W)의 행은 입력값(X)의 열크기와 동일하다.
W = tf.Variable(tf.random_normal([3, 1]), name='weight')
# Y 1개
b = tf.Variable(tf.random_normal([1]), name='bias')

# 이론(가설)
# X=5행3열 W=3행1열 => tf.matmul(X, W) => 결과=5행1열
hypothesis = tf.matmul(X, W) + b
cost = tf.reduce_mean(tf.square(hypothesis-Y))
# 최소 기울기
optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-5)
train = optimizer.minimize(cost)

# 실행
sess = tf.Session()
sess.run(tf.global_variables_initializer())

# 학습 횟수
for step in range(200001):
    cost_val, hy_val, _ = \
        sess.run([cost, hypothesis, train], feed_dict={X: x_data, Y: y_data})
    if step % 100 == 0:
        print(step, "cost:", cost_val, "\nprediction:\n", hy_val)


""" 
학습을 많이 시킬수록 비용이 떨어진다고 예측된다.

20000 cost: 1.2000163 
prediction:
 [[229.0054 ]
 [273.52908]
 [270.55215]
 [292.2802 ]
 [210.22676]]

200000 cost: 0.0005061333 
prediction:
 [[228.02727]
 [273.9847 ]
 [269.9787 ]
 [293.99014]
 [209.03166]]
"""

