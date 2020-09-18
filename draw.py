import matplotlib.pyplot as plt


def drawAll(data, points, parameters):
    '''
    输入:总体数据，关键点坐标，分段线性参数
    输出：显示数据，关键点，线性分段
    功能：绘制数据，关键点以及线性分段图形
    '''
    drawData(data)
    drawCriticalPoints(points)
    drawLinerFitting(parameters)


def drawData(data):
    plt.plot(range(len(data)), data, label='current_tem', linewidth=1)


def drawCriticalPoints(points):
    plt.scatter(points[0], points[1], marker='*', label='critical_points')


def drawLinerFitting(parameters):
    for k, b, s, e in parameters:
        x = range(s, e + 1)
        plt.plot(x, k * x + b, label='fitting', linewidth=1, color='red')
