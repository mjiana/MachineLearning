# 기울기
"""
최소 제곱법이란 회귀분석에서 사용되는 표준방식이다.
실험이나 관찰을 통해 얻은 데이터를 분석하여 미지의 상수를 구할 때 사용되는 공식이다.

최소 제곱법(최소자승법) : x와 y의 편차를 곱하여 이를 합한 값
=> 이 값이 기울기가 된다.

# 공식
시그마 (x-x평균)(y-y평균)
-------------------------- = 기울기 a
시그마 (x-x평균)^2
"""

import numpy as np  # 수치를 다루는 패키지

# 입력값 x 라벨값 y
x = [2, 4, 6, 8]
y = [81, 93, 91, 97]

# x와 y의 평균 구하기
mx = np.mean(x)
my = np.mean(y)
print('x의 평균값', mx)
print('y의 평균값', my)

# 기울기 공식의 분모
divisor = sum([(mx-i)**2 for i in x])


def top(x, mx, y, my):
    d = 0
    for i in range(len(x)):
        d += (x[i] - mx) * (y[i] - my)
    return d


dividend = top(x, mx, y, my)
print("분모", divisor)
print("분자", dividend)

# 기울기 a 절편 b
a = dividend/divisor
b = my - (mx*a)
print("기울기 a:", a)
print("절편 b:", b)

'''
x의 평균값 5.0
y의 평균값 90.5
분모 20.0
분자 46.0
기울기 a: 2.3
절편 b: 79.0
'''
