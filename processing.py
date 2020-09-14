import numpy as np
from typing import List


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
    distance = minDistance - 0.01*(maxFrontdistance + maxBackdistance)
    print((np.argmin(minDistance)))
    print(np.argmax(maxFrontdistance))
    print(np.argmax(maxBackdistance))
    return int(np.argmin(distance))
    # print("minDistance = ", minDistance)
    # print("maxFrontdistance", maxFrontdistance)
    # print("maxBackdistance", maxBackdistance)
