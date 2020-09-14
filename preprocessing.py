import datetime
import pandas as pd
import numpy as np
from typing import List

def minNums(startTime: str, endTime: str) -> int:
    '''计算两个时间点之间的分钟数'''
    startTime1 = datetime.datetime.strptime(startTime, r"%Y/%m/%d %H:%M:%S")
    endTime1 = datetime.datetime.strptime(endTime, r"%Y/%m/%d %H:%M:%S")
    # seconds = (endTime1 - startTime1).seconds
    # 来获取时间差中的秒数.seconds获得的秒只是时间差中的小时、分钟和秒部分的和，并没有包含时间差的天数（既是两个时间点不是同一天，失效）
    total_seconds = int((endTime1 - startTime1).total_seconds()
                        ) + startTime1.second - endTime1.second
    return total_seconds // 60

def getTime(startTime: str, num: int) -> str:
    startTime1 = datetime.datetime.strptime(startTime, r"%Y/%m/%d %H:%M:%S")
    time = startTime1 + datetime.timedelta(minutes=num)

    return time.strftime(r"%Y/%m/%d %H:%M:%S")

def interpolation(data: List[int]):
    ''' 窗口为3的平均插值函数'''
    size = len(data)
    for i in range(size):
        if data[i] == 0 and i > 0 and i < size - 1:
            data[i] = (data[i-1] + data[i+1]) // 2
        elif data[i] == 0 and i == size - 1:
            data[i] = data[i-1]
    return data


def load_csv_data(path: str) -> List[int]:
    corpus = pd.read_csv(
        path, usecols=["NOW_TIME", "CONTROLTEMP1_CURRENT_T"])

    
    # 转为list
    corpus = corpus.values.tolist()
    # print(corpus)
    start, end = corpus[0][0], corpus[-1][0]

    # 0-1有两个时刻，因此要+1
    size = minNums(start, end) + 1
    print('size = ', size)

    tdata = np.zeros(size, dtype='int')

    # 转换为从0时刻开始
    for c in corpus:
        tdata[minNums(start, c[0])] = c[1]

    # 插值，去掉0
    interpolation(tdata)

    return tdata
