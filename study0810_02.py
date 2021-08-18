#  softmax 결과들을 확률로 인식시키다.
# 예) 결과   5  7  3  1  ===>  0.2 + 0.4 + 0.3 + 0.1 = 1.0  로 결과를 표시함

#  onehot  가장 높은것은  1로 만들고  나머지는 0으로 처리하는것
#  예)  [ 0.2 , 0.4 , 0.3 , 0.1]  ====>  [ 0, 1 , 0 , 0 ]  하나만 두두러지게

#  argmax  가장 큰값의 위치를 반환한다.  [0,0,1] ==>  2  [0,1,0] ===> 1   [1,0,0] ===> 0

#  axis 축 : 1차 배열  0-열 , 2차 배열 0-행 1-열,  3차 배열 0-면 1-행 2-열 된다.

import tensorflow as tf
tf.set_random_seed(777)

x_data = [[1,2,1,1],
          [2,1,3,2],
          [3,1,3,4],
          [4,1,5,5],
          [1,7,5,5],
          [1,2,5,6],
          [1,6,6,6],
          [1,7,7,7]]
# argmax로 위치를 반환 2 2 2 1 1 1 0 0
y_data = [[0,0,1],  # onehot 형태로 표시
          [0,0,1],
          [0,0,1],
          [0,1,0],
          [0,1,0],
          [0,1,0],
          [1,0,0],
          [1,0,0]]

X = tf.placeholder("float",[None,4])
Y = tf.placeholder("float",[None,3])

nb_classes = 3

W = tf.Variable(tf.random_normal([4,nb_classes]),name='weight')
b = tf.Variable(tf.random_normal([nb_classes]),name='bias')

hypo = tf.nn.softmax(tf.matmul(X,W)+b)
cost = tf.reduce_mean(-tf.reduce_sum(Y*tf.log(hypo),axis=1))

train = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for step in range(2000):
        sess.run(train,feed_dict={X:x_data,Y:y_data})
        if step % 200 ==0 :
            print(step,sess.run(cost,feed_dict={X:x_data,Y:y_data}))

    # softmax로 얼마가 나온는지 확인하자
    a = sess.run(hypo,feed_dict={X:[[1,11,7,9]]})
    print("[[1,11,7,9]] softmax :",a,"  arg_max:",sess.run(tf.arg_max(a,1)))
    a = sess.run(hypo,feed_dict={X:[[1,1,0,1]]})
    print("[[1,1,0,1]] softmax :",a,"  arg_max:",sess.run(tf.arg_max(a,1)))
    a = sess.run(hypo,feed_dict={X:[[1,3,4,3]]})
    print("[[1,3,4,3]] softmax :",a,"  arg_max:",sess.run(tf.arg_max(a,1)))
















