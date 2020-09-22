import datetime
import pandas as pd
import numpy as np
from typing import List
startT = endT = None


def minNums(startTime, endTime) -> int:
    '''计算两个时间点之间的分钟数'''
    total_seconds = int((endTime - startTime).total_seconds()
                        ) + startTime.second - endTime.second
    return total_seconds // 60


def getTime(num: int, startTime) -> str:
    time = startTime + datetime.timedelta(minutes=num)
    return time


def avgInterpolation(data: List[int]):
    ''' 窗口为3的平均插值函数'''
    size = len(data)
    for i in range(size):
        if data[i] == 0 and i > 0 and i < size - 1:
            data[i] = (data[i-1] + data[i+1]) // 2
        elif data[i] == 0 and i == size - 1:
            data[i] = data[i-1]
    return data


def leftInterpolation(data: List[int]):
    size = len(data)
    for i in range(1, size):
        if data[i] == 0:
            data[i] = data[i-1]
    return data


def load_csv_data(path: str) -> List[int]:
    corpus = pd.read_csv(
        path, usecols=["NOW_TIME", "CONTROLTEMP1_CURRENT_T"])

    # 转为list
    corpus = corpus.values.tolist()

    start, end = corpus[0][0], corpus[-1][0]
    global startT, endT
    startT = datetime.datetime.strptime(start, r"%Y/%m/%d %H:%M:%S")
    endT = datetime.datetime.strptime(end,   r"%Y/%m/%d %H:%M:%S")

    # 0-1有两个时刻，因此要+1
    size = minNums(startT, endT) + 1
    print('size = ', size)

    tdata = np.zeros(size, dtype='int')

    # 转换为从0时刻开始
    for c in corpus:
        t = datetime.datetime.strptime(c[0], r"%Y/%m/%d %H:%M:%S")
        tdata[minNums(startT, t)] = c[1]

    # 插值，去掉0
    # avgInterpolation(tdata)
    leftInterpolation(tdata)

    return tdata
