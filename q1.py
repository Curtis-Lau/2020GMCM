import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot

df = pd.read_excel("q1285号.xlsx",index_col='时间')
df.index = pd.to_datetime(df.index)

df2 = pd.read_excel("q1313号.xlsx",index_col='时间')
df2.index = pd.to_datetime(df2.index)

df3 = pd.read_excel("附件一：325个样本数据.xlsx",index_col='样本编号')

# from pyod.models.knn import KNN   # imprt kNN分类器
# # 训练一个kNN检测器
# clf_name = 'kNN'
# clf = KNN() # 初始化检测器clf
# clf.fit(df2) # 使用X_train训练检测器clf

# # 返回训练数据X_train上的异常标签和异常分值
# y_train_pred = clf.labels_  # 返回训练数据上的分类标签 (0: 正常值, 1: 异常值)
# y_train_scores = clf.decision_scores_

# lst1 = ['S-ZORB.TE_1106.DACA.PV', 'S-ZORB.TE_1106.DACA']
# lst2 = ['S-ZORB.TE_1107.DACA.PV', 'S-ZORB.TE_1107.DACA']
# lst3 = ['S-ZORB.TE_1101.DACA.PV', 'S-ZORB.TE_1101.DACA']
# lst4 = ['S-ZORB.AT-0010.DACA.PV ', 'S-ZORB.AT-0011.DACA.PV']
#
# i = 1
# for l in [lst1,lst2,lst3,lst4]:
#     f,ax=pyplot.subplots(figsize=(8,6))
#     fig = sns.boxplot(data=df2.loc[:,l],order=l)
#     fig = fig.get_figure()
#     fig.savefig('q1pic{}.jpg'.format(i))
#     pyplot.show()
#     i += 1

def three_sigma(ser1):
    # 求平均值
    mean_value = ser1.mean()
    # 求标准差
    std_value = ser1.std()
    # 位于(μ-3σ,μ+3σ)区间内的数据是正常的，不在该区间的数据是异常的
    # ser1中的数值小于μ-3σ或大于μ+3σ均为异常值
    # 一旦发现异常值就标注为True，否则标注为False
    rule = (mean_value - 3 * std_value > ser1) | (mean_value + 3 * std_value < ser1)
    # 返回异常值的位置索引
    index = np.arange(ser1.shape[0])[rule]
    # 获取异常数据
    outrange = ser1.iloc[index]
    return(outrange)

lst = ['S-ZORB.FC_2801.PV','S-ZORB.PC_2105.PV','S-ZORB.PT_9402.PV','S-ZORB.FT_9401.PV','S-ZORB.AC_6001.PV',
       'S-ZORB.PC_1301.PV','S-ZORB.PT_1201.PV','S-ZORB.FC_1202.PV','S-ZORB.PC_1202.PV','S-ZORB.FC_3101.PV',
       'S-ZORB.PDC_2607.PV','S-ZORB.FT_1204.PV','S-ZORB.TE_1101.DACA','S-ZORB.TE_1107.DACA','S-ZORB.TE_1106.DACA',
       'S-ZORB.PC_3101.DACA','S-ZORB.LT_1501.DACA','S-ZORB.PDT_2503.DACA','S-ZORB.LT_1002.DACA','S-ZORB.FT_2431.DACA',
       'S-ZORB.PT_1101.DACA','S-ZORB.PT_6003.DACA','S-ZORB.FT_3702.DACA','S-ZORB.PT_2607.DACA','S-ZORB.PDT_2606.DACA',
       'S-ZORB.ZT_2634.DACA','S-ZORB.PC_2401B.PIDA.OP','S-ZORB.PDT_3502.DACA','S-ZORB.PT_2106.DACA','S-ZORB.PT_7510B.DACA',
       'S-ZORB.PT_7505B.DACA','S-ZORB.PT_7107B.DACA','S-ZORB.PT_1601.DACA','S-ZORB.TE_5008.DACA','S-ZORB.AT-0010.DACA.PV',
       'S-ZORB.AT-0011.DACA.PV','S-ZORB.AT-0013.DACA.PV','S-ZORB.PT_2106.DACA.PV','S-ZORB.FT_1204.DACA.PV','S-ZORB.TE_1106.DACA.PV',
       'S-ZORB.TE_1107.DACA.PV','S-ZORB.TE_1101.DACA.PV']

for col in lst:
    idx = list(three_sigma(df2[col]).index)
    mean_ = df2[col].mean()
    df2.loc[idx,col] = mean_

for col in list(df2.mean().index):
    df3.loc[313,col] = df2.mean()[col]

for col in list(df.mean().index):
    df3.loc[285,col] = df.mean()[col]

df3.to_excel('q1data.xlsx')