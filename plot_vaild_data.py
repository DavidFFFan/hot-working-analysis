import matplotlib.pyplot as plt
import csv

'''
根据曲线的角度变化找转折点，根据转折点划分趋势段
根据趋势段的角度变化和时间跨度进行合并，自底向上合并
'''

def readData(path):
    with open(path, mode='r') as f:
        reader = csv.reader(f)
        for data in reader:
            i_data = []
            for d in data:
                i_data.append(int(d))
            plt.plot(i_data)
    plt.show()
        
if __name__ == "__main__":
    readData("data/vaild_data.csv")