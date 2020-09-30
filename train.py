from sklearn import svm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import joblib

"""
采用 sklearn 中的svm 
"""
train_path = r'data\label\dataset11_3-11_19.csv'
test_path = r'data\label\dataset11_3-11_19.csv'


def load_data(data_path):
    '''
    X：abs(时间差，均值差，斜率)
    '''
    csv_reader = pd.read_csv(data_path)
    csv_reader.loc[:,"x1"] = abs(csv_reader['span'] - csv_reader['stdTime'])
    csv_reader.loc[:,"x2"] = abs(csv_reader['avg'] - csv_reader['stdTemperature'])
    csv_reader.loc[:,'x3'] = abs(csv_reader['k'])
    X = csv_reader[['x1','x2','x3']]
    Y = csv_reader.label
    return X, Y


def find_badcase(X, Y):
    bad_list = []
    y = cls.predict(X)
    for i in range(len(X)):
        if y[i] != Y[i]:
            bad_list.append(i)
    return bad_list


if __name__ == "__main__":
    # load data
    x_train, y_train = load_data(train_path)
    x_test, y_test = load_data(test_path)
    # training
    cls = svm.SVC(kernel='linear', C=1.5)
    cls.fit(x_train, y_train)
    joblib.dump(cls, "model_svm.joblib")
    # accuracy
    print('Test score: %.4f' % cls.score(x_test, y_test))
    print('Train score: %.4f' % cls.score(x_train, y_train))
    # print bad case id
    bad_idx = find_badcase(x_test, y_test)

    n_Support_vector = cls.n_support_  # 支持向量个数
    sv_idx = cls.support_  # 支持向量索引
    w = cls.coef_  # 方向向量W
    b = cls.intercept_

    # plot
    # 绘制分类平面
    ax = plt.subplot(111, projection='3d')
    x = np.arange(0, 500, 10)
    y = np.arange(0, 500, 10)
    x, y = np.meshgrid(x, y)
    z = (w[0, 0]*x + w[0, 1]*y + b) / (-w[0, 2])
    print(w)
    surf = ax.plot_surface(x, y, z, rstride=1, cstride=1)

    # 绘制三维散点图
    x_array = np.array(x_train, dtype=float)
    # print(x_array)
    y_array = np.array(y_train, dtype=int)
    pos = x_array[np.where(y_array == 1)]
    neg = x_array[np.where(y_array == 0)]
    ax.scatter(pos[:, 0], pos[:, 1], pos[:, 2], c='r', label='pos')
    ax.scatter(neg[:, 0], neg[:, 1], neg[:, 2], c='b', label='neg')

    # 绘制支持向量

    # X = np.array(x_train, dtype=float)
    # for i in range(len(sv_idx)):
    #     ax.scatter(X[sv_idx[i], 0], X[sv_idx[i], 1], X[sv_idx[i], 2], s=50,
    #               c='', marker='o', edgecolors='g')

    # 绘制 bad case
    # x_test = np.array(x_test,dtype=float)
    # for i in range(len(bad_idx)):
    #     j = bad_idx[i]
    #     ax.scatter(x_test[j,0], x_test[j,1], x_test[j,2],s=60,
    #                c='',marker='o', edgecolors='g')

    ax.set_zlabel('Z')    # 坐标轴
    ax.set_ylabel('Y')
    ax.set_xlabel('X')
    ax.set_zlim([0, 1])
    plt.legend(loc='upper left')

    ax.view_init(35, 300)
    plt.show()