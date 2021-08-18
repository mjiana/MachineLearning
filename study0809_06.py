# logistic regression : 로지스틱 회귀
# 결과가 0 또는 1, True 또는 False 로 회귀되는 경우
# ex 1) X 증상들을 나열하고, Y 발병 유무, 생존 유무를 판단하는 방식
# ex 2) 학습 시간을 주고, 합격 유무 판단
"""
Y는 0, 1로만 판단되어야 한다.
0과 1 두가지의 값 만으로는 기울기를 구할 수 없다.
따라서 0과 1사이를 부드러운 곡선으로 변환해서 기울기를 구해야한다.
이때 사용하는 함수가 sigmoid 이다.

비용함수는 log함수 그래프를 사용한다.
log함수의 장점 : 복잡한 계산을 간단히 처리할 수 있다.
=> sigmoid함수를 이용해서 비용곡선을 그리면 구불구불한 곡선으로 복잡하게 그려진다.
=> 이때 log함수를 사용해서 비용곡선을 그리면 선형함수 비용곡선 처럼 간단하게 그릴 수 있다.

즉 Y의 예측값이 0과 1사이에서 변할때 cost 비용의 변화를 나타내야 한다.
"""
import tensorflow as tf
tf.set_random_seed(777)

# 6 * 2
x_data = [[1, 2],
          [2, 3],
          [3, 1],
          [4, 3],
          [5, 3],
          [6, 2]]
# 6 * 1
y_data = [[0],
          [0],
          [0],
          [1],
          [1],
          [1]]

X = tf.placeholder(tf.float32, shape=[None, 2])
Y = tf.placeholder(tf.float32, shape=[None, 1])

W = tf.Variable(tf.random_normal([2, 1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

# sigmoid() : 계단형식의 함수를 미분이 가능하도록 곡선화 시켜주는 함수
# matmul() : 행렬의 곱셈을 구하는 함수
# log() : sigmoid()로인해 복잡해진 그래프를 간단하게 표현한다.
hypothesis = tf.sigmoid(tf.matmul(X, W)+b)
cost = -tf.reduce_mean(Y * tf.log(hypothesis) + (1-Y) * tf.log(1-hypothesis))
train = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)

# cast() : 정수일 때 절삭될 소수점 값들을 부동소수점 값으로 변경하고,
#          True/False 값을 숫자값으로 변경한다.
predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, Y), dtype=tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(10001):
        cost_val, _ = sess.run([cost, train], feed_dict={X: x_data, Y: y_data})
        if step % 200 == 0:
            print(step, cost_val)

    h, c, a = sess.run([hypothesis, predicted, accuracy],
                       feed_dict={X: x_data, Y: y_data})
    print("\nhypothesis:\n", h, "\nnCorrect(Y):\n", c, "\naccuracy:\n", a)

# 출력 결과
"""
9800 0.15177831
10000 0.14949557

hypothesis:
 [[0.03074028]
 [0.15884678]
 [0.30486736]
 [0.78138196]
 [0.93957496]
 [0.9801688 ]] 
nCorrect(Y):
 [[0.]
 [0.]
 [0.]
 [1.]
 [1.]
 [1.]] 
accuracy:
 1.0
"""
