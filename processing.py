import numpy as np
from typing import List
from sklearn import svm
import joblib
import math

def minEdistance(tdata: List[int], stdTime: int, stdTemp: int) -> (float, int):
    size, minDistance, minPos = len(tdata), float('inf'), -1

    # plt.plot(range(size), tdata, 'bo:', label='current_tem', linewidth=2)
    # plt.plot([stdTemp]*size,
    #          'yo:', label='std_tem', linewidth=2)

    # 第一个子串的距离
    distance = sum((tdata[:stdTime] -
                    stdTemp)**2)
    # print("minDistance = ",end=' ')
    # 求实际温度与标准工艺之间的最小距离，对于单一的温度工艺可以使用动态规划，时间复杂度O(m + n)
    for i in range(1, size - stdTime + 1):
        distance = distance - (tdata[i - 1] - stdTemp)**2 + (
            tdata[i - 1 + stdTime] - stdTemp)**2
        # print(distance,end=' ')
        if distance < minDistance:
            minDistance, minPos = distance, i
    # print()

    # print("minDistance", minDistance**0.5/stdTime, "minPos", minPos)
    return minDistance**0.5/stdTime, minPos
    # plt.show()


'''
def minmaxEdistance(tdata: List[int], stdTime: int, stdTemp: int):

    size, minPos = len(tdata), -1
    front = back = 5
    effectback = back if size - stdTime >= back else size - stdTime
    effectfront = front

    # Edistance 数组保存温度的欧氏距离
    Edistance = (tdata - stdTemp) ** 2
    preSumEdistance
    minDistance = np.zeros(size, dtype=int)
    maxBackdistance = np.zeros(size, dtype=int)
    maxFrontdistance = np.zeros(size, dtype=int)

    minDistance[0] = sum(Edistance[:stdTime])
    maxBackdistance[0] = sum(Edistance[stdTime:stdTime + back])
    for i in range(1, size - stdTime + 1):
        if effectfront > 0:
            effectfront -= 1
        minDistance[i] = minDistance[i - 1] - \
            Edistance[i - 1] + Edistance[i + stdTime - 1]
        maxFrontdistance[i] = maxFrontdistance[i - 1] + Edistance[i - 1]
        if i > front:
            maxFrontdistance[i] -= Edistance[i - front - 1]

        maxBackdistance[i] = maxBackdistance[i - 1] - \
            Edistance[i + stdTime - 1]
        if i + stdTime + back < size:
            maxBackdistance[i] += Edistance[i + stdTime + back - 1]

    print("minDistance = ", minDistance)
    print("maxFrontdistance", maxFrontdistance)
    print("maxBackdistance", maxBackdistance)
'''


def minmaxEdistance(tdata: List[int], stdTime: int, stdTemp: int):
    size, minPos = len(tdata), -1
    front = back = 5
    # Edistance 数组保存温度的欧氏距离，2阶距离
    # Edistance = (tdata - stdTemp) ** 2
    # 1阶距离
    Edistance = np.abs(tdata - stdTemp)
    preSumEdistance = np.zeros(size+1, dtype=int)
    for i in range(1, size + 1):
        preSumEdistance[i] = preSumEdistance[i - 1] + Edistance[i - 1]

    minDistance = np.zeros(size - stdTime + 1, dtype=int)
    maxBackdistance = np.zeros(size - stdTime + 1, dtype=int)
    maxFrontdistance = np.zeros(size - stdTime + 1, dtype=int)

    for i in range(0, size - stdTime + 1):
        minDistance[i] = preSumEdistance[i + stdTime] - preSumEdistance[i]
        maxFrontdistance[i] = preSumEdistance[i] - \
            preSumEdistance[i - front if i - front >= 0 else 0]
        maxBackdistance[i] = preSumEdistance[i+stdTime+back if i +
                                             stdTime+back <= size else size] - preSumEdistance[i+stdTime]
    # 距离标准化
    minDistance = np.sqrt(minDistance / stdTime)
    maxFrontdistance = np.sqrt(maxFrontdistance / front)
    maxBackdistance = np.sqrt(maxBackdistance / back)
    distance = minDistance - maxFrontdistance * maxBackdistance

    # distance = minDistance - 0.1*(maxFrontdistance * maxBackdistance)
    print((np.argmin(minDistance)))
    print(np.argmax(maxFrontdistance))
    print(np.argmax(maxBackdistance))
    minpos = int(np.argmin(distance))
    print("minpos = ", minpos)
    print("最小距离时前缀和为", maxFrontdistance[minpos], " ，后缀和为",
          maxBackdistance[minpos], "匹配度距离为", minDistance[minpos])
    return int(np.argmin(distance))


def findCriticalPoints(data):
    point = [[], []]
    THRESHOLD = 4
    for i in range(1, len(data) - 1):
        t = abs(data[i+1] - 2*data[i] + data[i-1])
        if t >= THRESHOLD:
            point[0].append(i)
            point[1].append(data[i])
    return point

def newfindCriticalPoints(data):
    point = [[], []]
    THRESHOLD = math.pi * 60 / 180
    for i in range(1, len(data) - 1):
        a1, a2 = math.atan(data[i] - data[i - 1]), math.atan(data[i + 1] - data[i])
        if a1 * a2 >= 0:
            t = abs(a1 - a2)
        else:
            t = abs(a1) + abs(a2)
        if t >= THRESHOLD:
            point[0].append(i)
            point[1].append(data[i])
    return point

def pieceswiseLinerFitting(data, xlabels):
    '''
    输入：总体数据， 关键点的横坐标
    输出：分段的线性拟合参数，包括斜率k,截距b，开始位置s,结束位置e
    '''
    parameters = []
    last, now = 0, None
    # 从起点到最后一个关键点
    for xlabel in xlabels:
        now = xlabel
        # 拟合区间左闭右闭[last, now]
        k, b = np.polyfit(range(last, now + 1), data[last:now + 1], 1)
        parameters.append((k, b, last, now))
        last = now
    # 最后一个关键点到数据结束
    now = len(data) - 1
    k, b = np.polyfit(range(last, now + 1), data[last:now + 1], 1)
    parameters.append((k, b, last, now))

    return parameters

def svmPredict(parameters, stdTime, stdTempure):
    '''
    使用SVM模型进行分类
    模型参数：abs(时间差，均值差，斜率)
    结果：1表示是，0不是
    '''
    # 加载SVM模型
    clf = joblib.load('model_svm.joblib')
    ans = []
    # X：abs(时间差，均值差，斜率)
    for (k, b, last, now) in parameters:
        if clf.predict([[abs((now-last) - stdTime), abs(k*(last+now)/2+b-stdTempure), abs(k)]])==1:
            # print('last = ',last,', now =',now)
            ans.append((k, b, last, now))
    return ans