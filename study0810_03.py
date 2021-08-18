# 데이터 차원을 맟주어 주는 것은 매우 중요하다.
# 데이터.shape : 차원을 출력해준다.
# reshape : 행열을 다시 조절함    reshape(2,3) 2행3열로 다시 만들어라
#          reshape(-1,7)  행은 상관없이 7열 짜리 모양으로 만들어라
# flatten  :  1차원 데이터로 전환     [[ ]]  2차 -----> [ ] 1차 데이터로 전환됨

# softmax_cross_entropy_with_logits() :  결과가 여러개로 분류되는 것의 비용함수

import tensorflow as tf
import numpy as np
tf.set_random_seed(777)

xy = np.loadtxt('./data/data-04-zoo.csv',delimiter=',',dtype=np.float32)
x_data = xy[:,0:-1]
y_data = xy[:,[-1]]
print(x_data.shape,y_data.shape )
# (101, 16) (101, 1)

nb_classes = 7  # 결과는 7개로 분류된다.
X = tf.placeholder(tf.float32,[None,16])
Y = tf.placeholder(tf.int32,[None,1])   # 정수값 0 ~ 6

Y_one_hot = tf.one_hot(Y,nb_classes)   #  4 ===> [0,0,0,1,0,0,0]
print("Y_one_hot:",Y_one_hot.shape)
# Y_one_hot (?, 1, 7)  3차배열
Y_one_hot = tf.reshape(Y_one_hot,[-1,nb_classes])
print("Y_reshape:",Y_one_hot.shape)
# Y_reshape: (?, 7)  2차배열

W = tf.Variable(tf.random_normal([16,nb_classes]),name='weight')
b = tf.Variable(tf.random_normal([nb_classes]),name='bias')
logits = tf.matmul(X,W)+b
hypo = tf.nn.softmax(logits)
#  softmax_cross_entropy_with_logits() :  결과가 여러개로 분류되는 것의 비용함수
cost_i = tf.nn.softmax_cross_entropy_with_logits(logits=logits,labels=Y_one_hot)

cost = tf.reduce_mean(cost_i)
opti = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)

# 예측값과 정확도
prediction = tf.arg_max(hypo,1)
correct_predition = tf.equal(prediction,tf.arg_max(Y_one_hot,1))
accuracy = tf.reduce_mean(tf.cast(correct_predition,tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for step in range(2000):
        sess.run(opti,feed_dict={X:x_data,Y:y_data})
        if step % 100 == 0 :
            loss,acc,cost_i_val = sess.run([cost,accuracy,cost_i],feed_dict={X:x_data,Y:y_data})
            print(step," loss:",loss,"  acc:",acc," cost_i:",cost_i_val.shape)
    pred = sess.run(prediction,feed_dict={X:x_data})
    for p,y in zip(pred,y_data.flatten()):
        print("[{}] predic {}  True Y:{} ".format(p==int(y),p,int(y)))

'''
1600  loss: 0.067723624   acc: 1.0
1700  loss: 0.063606516   acc: 1.0
1800  loss: 0.059974823   acc: 1.0
1900  loss: 0.056747712   acc: 1.0
중략 
True] predic 0  True Y:0 
[True] predic 1  True Y:1 
[True] predic 1  True Y:1 
[True] predic 0  True Y:0 
[True] predic 1  True Y:1 
[True] predic 5  True Y:5 
중략 
'''











