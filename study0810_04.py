# 정확도를 높이기 위하방법

# 1 학습데이터와 테스트 데이터를 분리한다.  6:4  또는 7:3 정도
# 2 leaning_rate 를 잘 적용해야 한다.
#    너무작으면  local minima 나 너무크면 overshooting 이 발생 할 수 있다.
import tensorflow as tf
tf.set_random_seed(777)

x_data =[[1,2,1],
         [1,3,2],
         [1,3,4],
         [1,5,5],
         [1,7,5],
         [1,2,5],
         [1,6,6],
         [1,7,7]]
y_data =[[0,0,1],
         [0,0,1],
         [0,0,1],
         [0,1,0],
         [0,1,0],
         [0,1,0],
         [1,0,0],
         [1,0,0]]
# 테스트 데이터 별도 준비 한다 .
x_test =[[2,1,1],
         [3,1,2],
         [3,3,4]]
y_test =[[0,0,1],
         [0,0,1],
         [0,0,1]]

X = tf.placeholder("float",[None,3])
Y = tf.placeholder("float",[None,3])

W = tf.Variable(tf.random_normal([3,3]))
b = tf.Variable(tf.random_normal([3]))

hypo = tf.nn.softmax(tf.matmul(X,W)+b)
# 크로스 엔트로피
# 비용 :
cost = tf.reduce_mean(-tf.reduce_sum(Y*tf.log(hypo),axis=1))
# 최적화 learning_rate 값에 따라서
# opti = tf.train.GradientDescentOptimizer(learning_rate=1e-10).minimize(cost) # 너무 작음   예측값이 안나옴
# opti = tf.train.GradientDescentOptimizer(learning_rate=2.5).minimize(cost)  #  너무 큼    nan 으로 출력됨
opti = tf.train.GradientDescentOptimizer(learning_rate=0.001).minimize(cost)  # 예측치가 나옴
# 예측치
predic = tf.arg_max(hypo,1)
# 일치
is_correct = tf.equal(predic,tf.arg_max(Y,1))
# 정확도
accu = tf.reduce_mean(tf.cast(is_correct,tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for step in range(2000):
        cost_val,W_val,_ = sess.run([cost,W,opti],feed_dict={X:x_data,Y:y_data})
        print("cost:",cost_val," W:",W_val)

    print("predic:",sess.run(predic,feed_dict={X:x_test}))
    print("accura:",sess.run(accu,feed_dict={X:x_test,Y:y_test}))





