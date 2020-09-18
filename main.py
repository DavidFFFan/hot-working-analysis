import os
import preprocessing as pp
from processing import *
import matplotlib.pyplot as plt
from draw import *

if __name__ == '__main__':
    path = r'./data/测试数据1天.csv'
    # 标准工艺的时间，以及温度
    time, temperature = 30, 815

    # 读取数据，数据预处理
    data = pp.load_csv_data(path)

    # 寻找关键点
    points = findCriticalPoints(data)

    # 分段线性拟合
    parameters = pieceswiseLinerFitting(data, points[0])

    # 画图
    drawAll(data, points, parameters)
    plt.show()

    # minDistance, minPos = minEdistance(tdata = data, stdTime= time, stdTemp=temperature)
    # print("minDistance", minDistance, "minPos", minPos)
