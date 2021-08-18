# 직접 손글씨 숫자그림을 작성하여 테스트하기
"""
1. 그림판을 이용하여 숫자 그림을 0~9까지 손글씨로 작성하고 저장
  * 저장위치 : 프로젝트 하단 data 폴더
2. 작성된 손글씨를 불러와서 확인
  * cv2 또는 opencv-python 패키지 설치
3. 저장된 학습결과 불러오기
4. 학습결과에 입력하려면 행렬이 일치해야 한다.
5. 학습결과에 1번에서 저장한 데이터를 제시하여 예측값 출력
"""
import cv2
from matplotlib import pyplot as plt
import numpy as np

# 이미지 불러오기
img = cv2.imread("data/9.png", cv2.IMREAD_GRAYSCALE)
# 이미지 가공 : 28*28사이즈의 흑백 전환
img = cv2.resize(255-img, (28, 28))

# 그림 확인
# plt.imshow(img)
# plt.show()

# 차원 확인
print("img 차원:", np.shape(img))  # img 차원: (28, 28)
# 1차원 데이터가 출력된다.
test_num = img.flatten()
# print(test_num)

# 우리가 필요한 데이터는 2차원 데이터 [1, 784]이다.
img = np.reshape(test_num, [1, 784])
print(img.shape)  # (1, 784)
# 이제 입력값으로 사용할 수 있는 img 가 생성되었다.

# 학습된 결과를 불러와서 사용하기
import tensorflow as tf
from matplotlib import pyplot as plt
tf.set_random_seed(777)

nb_classes = 10  # 0 ~ 9
# 28 * 28 = 784
X = tf.placeholder(tf.float32, [None, 784])
Y = tf.placeholder(tf.float32, [None, nb_classes])

W = tf.Variable(tf.random_normal([784, nb_classes]))
b = tf.Variable(tf.random_normal([nb_classes]))

hypothesis = tf.nn.softmax(tf.matmul(X, W) + b)

cost = tf.reduce_mean(-tf.reduce_sum(Y*tf.log(hypothesis), axis=1))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)

is_correct = tf.equal(tf.argmax(hypothesis, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))

save_file = "./model.ckpt"  # 학습 결과를 불러올 파일명
saver = tf.train.Saver()  # 저장객체 생성

# 저장된 파일을 세션에 다시 불러와서
# 학습과 반복 부분을 생략할 수 있다.
sess = tf.Session()
saver.restore(sess, save_file)

# 예측값 출력
print("Prediction:",
      sess.run(tf.argmax(hypothesis, 1), feed_dict={X: img}))
plt.imshow(
    img.reshape(28, 28),
    cmap="Greys",
    interpolation="nearest"
)
plt.show()

