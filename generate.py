import pandas as pd
import numpy as np
from preprocessing import getTime

def combine(name, parameters, startTime, stdTime, stdTemperature, nums):
    '''
    parameters为参数，分别为k, b, last, now
    startTime为开始时间
    stdtime为标准时间
    stdtemperature为标准温度
    nums为符合的个数
    '''
    # 开始时间
    df = pd.DataFrame({
        'startTime':pd.to_datetime([ getTime(x[2], startTime) for x in parameters]),
        'span':[x[3] - x[2] for x in parameters],
        'avg':[x[0]*(x[2]+x[3])/2+x[1] for x in parameters],
        'k':[x[0] for x in parameters],
        'stdTime':stdTime,
        'stdTemperature':stdTemperature,
        'label':pd.Series([1 if i < nums else 0 for i in range(len(parameters))], dtype='int')
    })
    print(df.dtypes)
    df.to_csv(name, index=False)