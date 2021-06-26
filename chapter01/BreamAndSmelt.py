import matplotlib.pyplot as plt

bream_length = [25.4, 26.3, 26.5, 29.0, 29.0, 29.7, 29.7, 30.0, 30.0, 30.7, 31.0, 31.0,
                31.5, 32.0, 32.0, 32.0, 33.0, 33.0, 33.5, 33.5, 34.0, 34.0, 34.5, 35.0,
                35.0, 35.0, 35.0, 36.0, 36.0, 37.0, 38.5, 38.5, 39.5, 41.0, 41.0]
bream_weight = [242.0, 290.0, 340.0, 363.0, 430.0, 450.0, 500.0, 390.0, 450.0, 500.0, 475.0, 500.0,
                500.0, 340.0, 600.0, 600.0, 700.0, 700.0, 610.0, 650.0, 575.0, 685.0, 620.0, 680.0,
                700.0, 725.0, 720.0, 714.0, 850.0, 1000.0, 920.0, 955.0, 925.0, 975.0, 950.0]

# feature가 두 개인 이차원 그래프
# 아래 그래프의 경우에는 일직선에 가까운 형태로 나타나는 linear 그래프.

plt.scatter(bream_length, bream_weight)
plt.xlabel("length")
plt.ylabel("weight")
# plt.show()

smelt_length = [9.8, 10.5, 10.6, 11.0, 11.2, 11.3, 11.8, 11.8, 12.0, 12.2, 12.4, 13.0, 14.3, 15.0]
smelt_weight = [6.7, 7.5, 7.0, 9.7, 9.8, 8.7, 10.0, 9.9, 9.8, 12.2, 13.4, 12.2, 19.7, 19.9]

plt.scatter(bream_length, bream_weight)
plt.scatter(smelt_length, smelt_weight)
# plt.show()

# 사용하는 algorithm k-nearest neighbors

# 두 개의 리스트를 이어 붙이는 작업 (각 값을 더하여 저장하는 기능이 아님을 유의)
length = bream_length + smelt_length
weight = bream_weight + smelt_weight

fish_data = [[l, w] for l, w in zip(length, weight)]
print(fish_data)

# 답안 리스트 (찾으려는 target의 경우 value를 1로, 그렇지 않으면 0이 관행)
fish_target = [1] * 35 + [0] * 14
print(fish_target)

from sklearn.neighbors import KNeighborsClassifier
# 사이킷런을 이용한 k-최근접 이웃 알고리즘 구현 클래스

kn = KNeighborsClassifier()
kn.fit(fish_data, fish_target)
# fit method는 주어진 데이터로 알고리즘을 훈련

print(kn.score(fish_data, fish_target))
# score method에서 출력되는 값은 accuracy 값을 의미

print(kn.predict([[30, 600]]))
# 이 값을 통하여 예측하면 value값을 1로 확인, 도미임을 확인

print(kn._fit_X)
# 전달한 fish 데이터
print(kn._y)
# 정답 데이터

# k-nearest neighbors algorithm은 무언가 훈련되는 게 없는 셈이다. fit method에서는 전달 데이터 저장만.

kn49 = KNeighborsClassifier(n_neighbors=49)
# default value = 5
# 대부분 도미(1) 값이므로 모든 케이스를 도미로 판단, 확률은 도미가 나올 확률, 따라서 35/49
kn49.fit(fish_data, fish_target)

print("bream_length " + str(len(bream_length)))
print("smelt_length " + str(len(smelt_length)))

print(kn49.score(fish_data, fish_target))
print(35/49)
