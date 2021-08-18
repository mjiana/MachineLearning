# RNN
import tensorflow as tf
import numpy as np

tf.set_random_seed(777)

sample = " if you want you"

# 리스트 형태, 중복불가, 정렬이 없어 매번 변경된다.
idx2char = list(set(sample))  # [' ', 'i', 'w', 't', 'u', 'o', 'y', 'n', 'a', 'f']
# 딕셔너리 형태로 변경
char2idx = {c: i for i, c in enumerate(idx2char)}
# {'o': 0, 'y': 1, 'u': 2, 't': 3, 'w': 4, 'i': 5, 'n': 6, ' ': 7, 'f': 8, 'a': 9}

# 입력 문자열에 따라 자동으로 각 크기 결정
# len(char2idx) : 10개 : 구성요소의 개수
# 열 크기
dic_size = len(char2idx)  # 사전크기
hidden_size = len(char2idx)  # 은닉층 출력개수
num_classes = len(char2idx)  # 최종 출력결과 개수
# 행 크기
sequence_length = len(sample) - 1  # 16-1 = 15개
# 면 크기
batch_size = 1

learning_rate = 0.1

sample_idx = [char2idx[c] for c in sample]  # 16개 : 전체 개수

x_data = [sample_idx[:-1]]  # 15개 : 뒤에서 한개 제외
y_data = [sample_idx[1:]]  # 15개 : 앞에서 한개 제외

# 행무시, 15열 / 2차원
X = tf.placeholder(tf.int32, [None, sequence_length])
Y = tf.placeholder(tf.int32, [None, sequence_length])
# 3차원으로 변경되어 최종 학습 데이터는 1면 15행 10열이 된다.
x_one_hot = tf.one_hot(X, num_classes)

# 은닉층 정의
cell = tf.contrib.rnn.BasicLSTMCell(num_units=hidden_size, state_is_tuple=True)
# 은닉층의 초기상태는 값이 없으므로 0.0
initial_state = cell.zero_state(batch_size, tf.float32)
# 입력값 X를 가지고 은닉층을 얼마나 펼쳐야할지 결과 구하기
outputs, _states = tf.nn.dynamic_rnn(
    cell, x_one_hot, initial_state=initial_state, dtype=tf.float32)

# 출력층
# fully_connected 를 출력값에 전달하기 위해 모양 변형
# 행 무시, 10열 결과
X_for_fc = tf.reshape(outputs, [-1, hidden_size])
# 최종 학습 데이터 1면 15행 10열으로 변경
outputs = \
    tf.contrib.layers.fully_connected(outputs, num_classes, activation_fn=None)
outputs = tf.reshape(outputs, [batch_size, sequence_length, num_classes])
# 가중치 1 * 15, 값은 모두 1
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
            l, _ = sess.run([loss, train], feed_dict={X: x_data, Y: y_data})
            result = sess.run(prediction, feed_dict={X: x_data})
            # 1차원으로 변형, 사전에서 선택
            result_str = [idx2char[c] for c in np.squeeze(result)]

            print(i, "loss:", l, "prediction:", ''.join(result_str))

''' 실행 결과
0 loss: 2.310959 prediction:                
200 loss: 2.0174642 prediction: u   ou u     ou
(생략)
9900 loss: 0.00035992457 prediction: if you want you
'''
