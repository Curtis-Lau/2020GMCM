import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def GM11(x, n):
    '''
    灰色预测
    x：序列，numpy对象
    n:需要往后预测的个数
    '''
    x1 = x.cumsum()  # 一次累加
    z1 = (x1[:len(x1) - 1] + x1[1:]) / 2.0  # 紧邻均值
    z1 = z1.reshape((len(z1), 1))
    B = np.append(-z1, np.ones_like(z1), axis=1)
    Y = x[1:].reshape((len(x) - 1, 1))
    # a为发展系数 b为灰色作用量
    [[a], [b]] = np.dot(np.dot(np.linalg.inv(np.dot(B.T, B)), B.T), Y)  # 计算参数
    result = (x[0] - b / a) * np.exp(-a * (n - 1)) - (x[0] - b / a) * np.exp(-a * (n - 2))

    S1_2 = x.var()  # 原序列方差
    e = list()  # 残差序列
    for index in range(1, x.shape[0] + 1):
        predict = (x[0] - b / a) * np.exp(-a * (index - 1)) - (x[0] - b / a) * np.exp(-a * (index - 2))
        e.append(x[index - 1] - predict)
    S2_2 = np.array(e).var()  # 残差方差
    C = S2_2 / S1_2  # 后验差比
    if C <= 0.35:
        assess = '后验差比<=0.35，模型精度等级为一级'
    elif C <= 0.5:
        assess = '后验差比<=0.5，模型精度等级为二级'
    elif C <= 0.65:
        assess = '后验差比<=0.65，模型精度等级为三级'
    else:
        assess = '后验差比>0.65，模型精度等级为四级'
    # 预测数据
    predict = list()
    for index in range(x.shape[0] + 1, x.shape[0] + n + 1):
        predict.append((x[0] - b / a) * np.exp(-a * (index - 1)) - (x[0] - b / a) * np.exp(-a * (index - 2)))
    predict = np.array(predict)
    return {
        'a': {'value': a, 'desc': '发展系数'},
        'b': {'value': b, 'desc': '灰色作用量'},
        'predict': {'value': result, 'desc': '第%d个预测值' % n},
        'C': {'value': C, 'desc': assess},
        'predict': {'value': predict, 'desc': '往后预测%d个的序列' % (n)},
    }

if __name__ == "__main__":
    data = np.array([3664.97,3725.66,3792.49,3865.14,3962.42,4085.74,4183.93,4202.66,4212.50,4223.95,4206.66])
    x = data[:]  # 输入数据
    # y = data[4:]  # 需要预测的数据
    result = GM11(x, 2)
    # predict = result['predict']['value']
    # accuracy = result['C']['desc']
    # predict = np.round(predict, 3)
    # print('真实值:', y)
    # print('预测值:', predict)
    print(result)


# 用来正常显示中文标签
# plt.rcParams['font.sans-serif']=['SimHei']
# 用来正常显示负号
# plt.rcParams['axes.unicode_minus']=False
# plt.plot(range(test.shape[0]),yPre,label="预测值")
# plt.plot(range(test.shape[0]),test,label="观测值")
# plt.legend()
# plt.title('GM11预测效果，MAE：%2f'%MAE)
# plt.savefig(resultDir+'/GM11预测效果.png',dpi=100,bbox_inches='tight')