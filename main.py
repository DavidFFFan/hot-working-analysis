import os
import preprocessing as pp
from processing import *
import matplotlib.pyplot as plt



if __name__ == '__main__':
    path = r'./data/测试数据1天.csv'
    # 标准工艺的时间，以及温度
    time, temperature = 30, 815

    data = pp.load_csv_data(path)
    
    # 打印实际数据曲线
    plt.plot(range(len(data)), data, 'bo:', label='current_tem', linewidth=1)
    
    # minDistance, minPos = minEdistance(tdata = data, stdTime= time, stdTemp=temperature)
    # print("minDistance", minDistance, "minPos", minPos)
    print( pp.getTime(minmaxEdistance(data, time, temperature), pp.startT))

    plt.show()
