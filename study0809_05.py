# 데이터 파일의 크기가 크거나 여러개인 경우
import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.set_random_seed(777)

filename_queue = tf.train.string_input_producer(
        ['./data/data-01-test-score.csv'], shuffle=False, name='filename_queue')

reader = tf.TextLineReader()  # reader 객체 생성
key, value = reader.read(filename_queue)

# print(key, value)
''' 출력 결과
Tensor("ReaderReadV2:0", shape=(), dtype=string) 
Tensor("ReaderReadV2:1", shape=(), dtype=string)
'''

recode_default = [[0.], [0.], [0.], [0.]]
xy = tf.decode_csv(value, record_defaults=recode_default)
# print(xy)
''' 출력 결과
[<tf.Tensor 'DecodeCSV:0' shape=() dtype=float32>, 
<tf.Tensor 'DecodeCSV:1' shape=() dtype=float32>, 
<tf.Tensor 'DecodeCSV:2' shape=() dtype=float32>, 
<tf.Tensor 'DecodeCSV:3' shape=() dtype=float32>]
'''

# 데이터 슬라이싱
train_x_batch, train_y_batch = tf.train.batch([xy[0:-1], xy[-1:]], batch_size=10)
# print(train_x_batch, train_y_batch)
''' 출력 결과
Tensor("batch:0", shape=(10, 3), dtype=float32) 
Tensor("batch:1", shape=(10, 1), dtype=float32)
'''
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
# 여러개의 파일을 읽어올 때 코디네이터 처리
coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=sess, coord=coord)

for step in range(10000):
    x_batch, y_batch = sess.run([train_x_batch, train_y_batch])
    cost_val, hy_val, _ = \
        sess.run([cost, hypothesis, train], feed_dict={X: x_batch, Y: y_batch})
    if step % 1000 == 0:
        print("step:", step)

coord.request_stop()
coord.join(threads)

# 테스트 - 예측이 될까?
print("[99,99,99] 예측값은?:",
      sess.run(hypothesis, feed_dict={X: [[99, 99, 99]]}))
# 결과 : [[199.96214]]
print("[88, 77, 66], [91, 81, 61] 예측값은?:",
      sess.run(hypothesis, feed_dict={X: [[88, 77, 66], [91, 81, 61]]}))
# 결과 : [[147.14406] [144.8244 ]]


