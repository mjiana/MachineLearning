# report
# 교안 12-5 3454줄
# 강사님 파일은 study0817_04.py로 되어있다.
"""
Open 시가(개장), High 최고가, Low 최저가, Volume 거래량, Close 종가(폐장)
7일간의 데이터
입력 5개
출력 1개
은닉 10
csv파일 읽기
종가 제외
5개 중에 가장 마지막 값이 Y-라벨값
0부터 Y의 길이 - 기간만큼 반복문
"""
# 라이브러리 참조
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# 일정한 범위에서 랜덤으로
tf.set_random_seed(777)


# 함수
def MinMaxScaler(data):
    numerator = data - np.min(data, 0)
    denominator = np.max(data, 0) - np.min(data, 0)
    return numerator / (denominator+1e-7)


# train parameters
seq_length = 7  # 7일 동안
data_dim = 5  # 입력 데이터 5개(개장, 최고, 최저, 거래량, 폐장)
hidden_dim = 10  # 은닉층 10개 확장
output_dim = 1  # 출력 데이터 1개
learning_rate = 0.01
iterations = 500  # 반복

# 데이터 전처리
xy = np.loadtxt("./data/data-02-stock_daily.csv", delimiter=",")
xy = xy[::-1]  # 행을 역순으로 재정렬

# 데이터 값이 너무 크면 학습이 불가능하다.
xy = MinMaxScaler(xy)  # 0~1 사이로 변환, 학습효과 상승

x = xy
print("x:", x)
'''
[ (생략)
 [9.56900354e-01 9.59881106e-01 9.80354497e-01 1.42502465e-01
  9.77850239e-01]
 [9.73335806e-01 9.75431522e-01 1.00000000e+00 1.11123062e-01
  9.88313020e-01]]
'''
# 데이터 슬라이싱 : 마지막 열만 선택
y = xy[:, [-1]]
print("y:", y)
'''
[ (생략)
 [0.97785024]
 [0.98831302]]
'''

# 데이터셋 만들기
dataX = []
dataY = []
for i in range(0, len(y)-seq_length):  # 0 ~ (732 - 7 = 725)
    _x = x[i:i+seq_length]
    _y = y[i+seq_length]
    dataX.append(_x)
    dataY.append(_y)
    print("입력 데이터:", _x, "-----> 라벨:", _y)
''' 
입력 데이터: 7행 5열
[[0.91753068 0.90955899 0.93013248 0.08799857 0.92390372]
 [0.92391259 0.92282604 0.94550876 0.10049296 0.93588207]
 [0.93644323 0.93932734 0.96226394 0.10667742 0.95211558]
 [0.94518557 0.94522671 0.96376051 0.09372591 0.95564213]
 [0.9462346  0.94522671 0.97100833 0.11616922 0.9513578 ]
 [0.94789567 0.94927335 0.97250489 0.11417048 0.96645463]
 [0.95690035 0.95988111 0.9803545  0.14250246 0.97785024]] 
 -----> 라벨: 1행1열 [0.98831302]
'''

# 연습데이터와 학습 데이터 분리
train_size = int(len(dataY) * 0.7)  # 학습데이터는 70%만 사용하고
test_size = len(dataY) - train_size  # 나머지를 테스트 사이즈로 사용

trainX, testX = np.array(dataX[0:train_size]), \
                np.array(dataX[train_size:len(dataX)])

trainY, testY = np.array(dataY[0:train_size]), \
                np.array(dataY[train_size:len(dataY)])

# 5종 자료가 7일간 들어가서 n면 7행 5열이 되고 1개가 출력된다.
X = tf.placeholder(tf.float32, [None, seq_length, data_dim])
Y = tf.placeholder(tf.float32, [None, 1])

# 은닉층
cell = tf.contrib.rnn.BasicLSTMCell(
            num_units=hidden_dim, state_is_tuple=True, activation=tf.tanh)
# 다음 은닉층으로 전달
outputs, _states = tf.nn.dynamic_rnn(cell, X, dtype=tf.float32)
# 출력층은 마지막 값만 사용
Y_pred = tf.contrib.layers.fully_connected(
            outputs[:, -1], output_dim, activation_fn=None)

# 비용 손실 : 오차제곱평균
loss = tf.reduce_sum(tf.square(Y_pred - Y))
# 최적화
optimizer = tf.train.AdamOptimizer(learning_rate)
train = optimizer.minimize(loss)

# RMSE : 오차제곱평균근
targets = tf.placeholder(tf.float32, [None, 1])
predictions = tf.placeholder(tf.float32, [None, 1])
rmse = tf.sqrt(tf.reduce_mean(tf.square(targets-predictions)))

# 실행
with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)

    # 학습
    for i in range(iterations):
        _, step_loss = sess.run([train, loss], feed_dict={X: trainX, Y: trainY})
        print("[step:{}] loss: {}".format(i, step_loss))

    # 테스트
    test_predict = sess.run(Y_pred, feed_dict={X: testX})
    rmse_val = sess.run(rmse, feed_dict={targets: testY, predictions: test_predict})
    print("RMSE: {}".format(rmse_val))

    plt.plot(testY)  # 테스트 라벨값
    plt.plot(test_predict)  # 예측 값
    plt.xlabel("Time Period")  # x축의 라벨명
    plt.ylabel("Stock Price")  # y축의 라벨명
    plt.show()  # 시각화










