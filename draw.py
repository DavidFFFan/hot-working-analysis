import matplotlib.pyplot as plt
import csv

def drawAll(data, points, parameters):
    '''
    输入:总体数据，关键点坐标，分段线性参数
    输出：显示数据，关键点，线性分段
    功能：绘制数据，关键点以及线性分段图形
    '''
    plt.xlabel('time')
    plt.ylabel('temperature')
    drawData(data)
    drawCriticalPoints(points)
    drawLinerFitting(parameters)
    
    '''
    # 提取数据
    with open("data/vaild_data.csv", mode='w') as f:
        writer = csv.writer(f)
        for k, b, s, e in parameters:
            p_data = data[s - 80: e + 80]
            writer.writerow(p_data)
    '''

def drawData(data):
    plt.plot(range(len(data)), data, label='current_tem', linewidth=1)


def drawCriticalPoints(points):
    plt.scatter(points[0], points[1], marker='*', label='critical_points')


def drawLinerFitting(parameters):
    flag = True
    for k, b, s, e in parameters:
        x = range(s, e + 1)
        if flag:
            plt.plot(x, k * x + b, linewidth=1, label='fitting', color='red')
            flag = False
        else:
            plt.plot(x, k * x + b, linewidth=1, color='red')
