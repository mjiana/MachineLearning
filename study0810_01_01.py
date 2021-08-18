# 당뇨병 진단하기
# logistic regression 로지스틱 회귀 ,     결과가 0 이나 1로 귀결되는 경우 ,참거짓 , yes no
import tensorflow as tf
import numpy as np
tf.set_random_seed(777)

xy = np.loadtxt('./data/data-03-diabetes.csv',delimiter=',',dtype=np.float32)
x_data = xy[:,0:-1]
y_data = xy[:,[-1]]

X = tf.placeholder(tf.float32,shape=[None,8])  # 행무관 2열데이터 입력
Y = tf.placeholder(tf.float32,shape=[None,1])  # 행무관 2열데이터 입력

W = tf.Variable(tf.random_normal([8,1]),name='weight')
b = tf.Variable(tf.random_normal([1]),name='bias')

hypo = tf.sigmoid(tf.matmul(X,W)+b)

cost = -tf.reduce_mean(Y*tf.log(hypo)+(1-Y)*tf.log(1-hypo))

train = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)

predicted = tf.cast(hypo > 0.5,dtype=tf.float32) # 1.0  0.0
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted,Y),dtype=tf.float32))

with tf.Session() as sess :
    sess.run(tf.global_variables_initializer())

    for step in range(10000):
        cost_val,_=sess.run([cost,train],feed_dict={X:x_data,Y:y_data})
        if step % 200 == 0:
            print(step," cost:",cost_val)

    h, c, a = sess.run([hypo,predicted,accuracy],feed_dict={X:x_data,Y:y_data})
    print("Hypo:",h," predicted:",c," accuracy:", a)

'''
 중략 ~~~~~
9200  cost: 0.49348038
9400  cost: 0.49275038
9600  cost: 0.49205622
9800  cost: 0.49139577
Hypo: [[0.44349048]
 중략 ~~~~~
 [0.7461004 ]
 [0.7991883 ]
 [0.72995377]
 [0.88296574]]  predicted: [[0.]
 중략 ~~~~~
[0.]
 [1.]
 [1.]
 [1.]
 [1.]
 [1.]
 [1.]]  accuracy: 0.7628459
'''


import random as rd
import numpy as np

# 8 * 5 2차 배열
xy = []
for i in range(40):
    xy.append(rd.uniform(800, 1000))  # 800~1000사이의 실수인 난수가 발생한다.
print(xy)  # 1차 배열
print(np.reshape(xy, [8, 5]))  # 2차 배열
xy = np.reshape(xy, [8, 5])


def MinMaxScalar(data):
    up = data-np.min(data, 0)
    down = np.max(data, 0) - np.min(data, 0)
    return up/(down+1e-5)

xy = MinMaxScalar(xy)
print(xy)