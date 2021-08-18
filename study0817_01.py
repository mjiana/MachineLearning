# RNN
# 교재 12-1, 3187줄
"""
RNN : Recurrent Neural Network : 순환 신경망
상세내용은 study0817_01.txt 참조
"""
import tensorflow as tf
import numpy as np

tf.set_random_seed(777)

# 5열 : 사전  0    1    2    3    4
idx2char = ['h', 'i', 'e', 'l', 'o']
# 1행 6열 : 입력데이터 hihell
x_data = [[0, 1, 0, 2, 3, 3]]
# 1면 6행 5열 <= 실제 학습 방식
x_one_hot = [[[1, 0, 0, 0, 0],
             [0, 1, 0, 0, 0],
             [1, 0, 0, 0, 0],
             [0, 0, 1, 0, 0],
             [0, 0, 0, 1, 0],
             [0, 0, 0, 1, 0]]]
# 1행 6열 : 출력데이터 ihello
y_data = [[1, 0, 2, 3, 3, 4]]

num_classes = 5  # 출력결과 개수 : idx2char 값 중 하나
hidden_size = 5  # num_classes 와 동일
input_dim = 5  # 열의 입력크기 : x_one_hot의 크기
sequence_length = 6  # 행의 크기 : 입력단어의 길이 hihell
batch_size = 1   # 면의 크기 : 한번 입력되는 단어 hihell 
# => 단어가 if you want 인 경우는 3이 된다.
learning_rate = 0.1

X = tf.placeholder(tf.float32, [None, sequence_length, input_dim])
Y = tf.placeholder(tf.int32, [None, sequence_length])

# 은닉층 정의
cell = tf.contrib.rnn.BasicLSTMCell(num_units=hidden_size, state_is_tuple=True)
# 은닉층의 초기상태는 값이 없으므로 0.0
initial_state = cell.zero_state(batch_size, tf.float32)
# 입력값 X를 가지고 은닉층을 얼마나 펼쳐야할지 결과 구하기
outputs, _states = tf.nn.dynamic_rnn(
    cell, X, initial_state=initial_state, dtype=tf.float32)

# 출력층
# fully_connected 를 출력값에 전달하기 위해 모양 변형
# 행 무시, 5열 결과
X_for_fc = tf.reshape(outputs, [-1, hidden_size])
outputs = tf.contrib.layers.fully_connected(
            inputs=X_for_fc, num_outputs=num_classes, activation_fn=None)
# 1면 6행 5열 모양으로 변경
outputs = tf.reshape(outputs, [batch_size, sequence_length, num_classes])
# 가중치 모양(1*6) 값은 모두 1
weights = tf.ones([batch_size, sequence_length])

# 연속 손실, 연속 비용
sequence_loss = tf.contrib.seq2seq.sequence_loss(
    logits=outputs, targets=Y, weights=weights)
# 평균 비용
loss = tf.reduce_mean(sequence_loss)
# 학습노드
train = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

# 예측 : 열 기준 최대값의 위치 반환
prediction = tf.argmax(outputs, axis=2)

# 실행
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(10000):
        if i % 100 == 0:
            l, _ = sess.run([loss, train], feed_dict={X: x_one_hot, Y: y_data})
            result = sess.run(prediction, feed_dict={X: x_one_hot})
            print(i, "loss:", l, "prediction:", result, "true Y:", y_data)
            # 1차원으로 변형, 사전에서 선택
            result_str = [idx2char[c] for c in np.squeeze(result)]
            print("prediction str:", ''.join(result_str))

''' 실행 결과
0 loss: 1.6231518 prediction: [[3 3 3 3 3 3]] true Y: [[1, 0, 2, 3, 3, 4]]
prediction str: llllll
(생략)
9900 loss: 0.0013295076 prediction: [[1 0 2 3 3 4]] true Y: [[1, 0, 2, 3, 3, 4]]
prediction str: ihello
'''

