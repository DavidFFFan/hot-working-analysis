import os
import preprocessing as pp
from processing import *
import matplotlib.pyplot as plt
from draw import *
from generate import combine

# def func(x):
#     return (30*abs(x[3] - x[2] - time)/time+ abs(x[0]*(x[2]+x[3])/2+x[1]-temperature) + abs(x[0]))
if __name__ == '__main__':
    pathIn = r'data/origin/测试数据11_3-11_19.csv'
    pathOut = r'data/label/dataset11_3-11_19.csv'
    # 标准工艺的时间，以及温度
    time, temperature = 30, 815

    # 读取数据，数据预处理
    data = pp.load_csv_data(pathIn)

    # 寻找关键点
    points = newfindCriticalPoints(data)

    # 分段线性拟合
    parameters = pieceswiseLinerFitting(data, points[0])

    # t = sorted(parameters, key=func)
    # SVM预测
    t = svmPredict(parameters, time, temperature)

    # 画图
    drawAll(data, points, t)
    
    # combine(pathOut,t, pp.startT, time, temperature, 8)
    plt.legend()
    plt.show()

    # minDistance, minPos = minEdistance(tdata = data, stdTime= time, stdTemp=temperature)
    # print("minDistance", minDistance, "minPos", minPos)
