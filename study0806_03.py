# 비용함수 도출 과정

import numpy as np

# 기울기 a y절편 b
ab = [3, 76]

# x, y 데이터값
# x: 공부한 시간, y: 획득점수
data = [[2, 81], [4, 93], [6, 91], [8, 97]]
'''
실제 모양
0열 1열
[2, 81]
[4, 93]
[6, 91]
[8, 97]
'''
x = [i[0] for i in data]  # 0열의 데이터
# print(x)
y = [i[1] for i in data]  # 1열의 데이터


# 이론공식 함수화
def predict(x):
    return ab[0] * x + ab[1]

# RMSE : Root Mean Squared Error : 오차 제곱 평균 근 구하기 함수
# 장점 : 지표자체가 직관적이고 단순하다.
# 단점 :
# 스케일 의존적 - 비교대상의 수치 차이가 크지만 오차크키만 가지고 비교하기 때문에 차이가 없어보인다.
# ex) A주가 1,000,000 / B주가 200,000 일때 오차가 50,000인 경우
#     오차만 봤을 때 A와 B의 RMSE가 같아보인다.
# 왜곡 - 에러를 제곱하기 때문에 1미만의 에러는 적어지고 그 이상의 에러는 더 커진다.
def rmse(p, a):
    return np.sqrt(((p-a)**2).mean())


# RMSE 함수 대입 결과 도출
def rmse_val(predict_result, y):
    return rmse(np.array(predict_result), np.array(y))


predict_result = []
for i in range(len(x)):
    predict_result.append(predict(x[i]))
    print("공부한 시간=%f, 실제 점수=%f, 예측점수=%f" % (x[i], y[i], predict(x[i])))

# 오차 제곱 평균 근 
print("RMSE 최종값 :", str(rmse_val(predict_result, y)))

'''
예측점수 : 직접 설정한 이론(y=x*W+b)에 의해 기대되는 점수
공부한 시간=2.000000, 실제 점수=81.000000, 예측점수=82.000000
공부한 시간=4.000000, 실제 점수=93.000000, 예측점수=88.000000
공부한 시간=6.000000, 실제 점수=91.000000, 예측점수=94.000000
공부한 시간=8.000000, 실제 점수=97.000000, 예측점수=100.000000
RMSE 최종값 : 3.3166247903554
'''

