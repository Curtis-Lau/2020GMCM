from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import numpy as np

df = pd.read_excel('q1data.xlsx',index_col='样本编号')
df_y = df['辛烷值RON.1']
df_x = df.drop(['时间','RON损失','辛烷值RON.1','辛烷值RON'],axis=1)

#构造自变量对应数据集
nrows = df_x.shape[0]
data = []
for i in range(0, nrows):
    data.append(list(df_x.iloc[i]))
names = list(df_x.columns)

#构造目标变量对应的数据集
target = df_y.values

#构造随机森林算法
X = data
Y = target
rf = RandomForestRegressor()
rf.fit(df_x, df_y)
print("Features sorted by their score:")
a = sorted(zip(map(lambda x: round(x, 4), rf.feature_importances_), names), reverse=True)
# print(a)
# col1是随机森林选出的前20个特征
col1 = [i[1] for i in a[:20]]

from numpy import array
from sklearn.feature_selection import SelectKBest
from minepy import MINE
from scipy.stats import pearsonr

# 互信息发
# 由于MINE的设计不是函数式的，定义mic方法将其为函数式的，返回一个二元组，二元组的第2项设置成固定的P值0.5
def mic(x, y):
    m = MINE()
    m.compute_score(x, y)
    return (m.mic(), 0.5)

# 选择K个最好的特征，返回特征选择后的数据
mat = SelectKBest(lambda X, Y: list(array([pearsonr(x, Y) for x in X.T]).T), k=60).fit_transform(X,Y)
df_col2 = pd.DataFrame(mat)
col2 = ['辛烷值RON','硫含量,μg/g','烯烃,v%','密度(20℃),\nkg/m³','S-ZORB.SIS_PDT_2103B.PV','S-ZORB.LC_5101.PV','S-ZORB.FT_5104.PV',
        'S-ZORB.FT_1001.PV','S-ZORB.FT_1004.PV','S-ZORB.SIS_LT_1001.PV','S-ZORB.TE_1001.PV','S-ZORB.AT_1001.PV',
        'S-ZORB.TE_1105.PV','S-ZORB.TE_1201.PV','S-ZORB.FC_1203.PV','S-ZORB.FT_9101.TOTAL','S-ZORB.FT_9201.TOTAL',
        'S-ZORB.FT_9301.TOTAL','S-ZORB.FT_9402.TOTAL','S-ZORB.FT_5201.PV','S-ZORB.TE_1101.DACA','S-ZORB.TE_1106.DACA',
        'S-ZORB.FT_3303.DACA','S-ZORB.SIS_PT_2703','S-ZORB.SIS_TE_2802','S-ZORB.TE_5002.DACA','S-ZORB.FT_3201.DACA',
        'S-ZORB.AT_1001.DACA','S-ZORB.PDI_2903.DACA','S-ZORB.PT_7510.DACA','S-ZORB.PT_7107.DACA','S-ZORB.PT_7103.DACA',
        'S-ZORB.TE_7102.DACA','S-ZORB.SIS_FT_3202.PV','S-ZORB.TE_3112.DACA','S-ZORB.PT_2106.DACA.PV','S-ZORB.TE_6008.DACA.PV',
        'S-ZORB.TE_6001.DACA.PV','S-ZORB.TE_1104.DACA.PV','S-ZORB.TE_1101.DACA.PV','S-ZORB.FT_5204.TOTALIZERA.PV',
        '硫含量,μg/g.1','S-ZORB.FT_9403.PV','S-ZORB.FT_9402.PV','S-ZORB.FT_5201.TOTAL','S-ZORB.FT_5101.TOTAL',
        'S-ZORB.FT_1003.TOTAL','S-ZORB.FT_9202.TOTAL','S-ZORB.FT_9302.TOTAL','S-ZORB.FT_9401.TOTAL','S-ZORB.FT_9403.TOTAL',
        'S-ZORB.FC_1101.TOTAL','S-ZORB.FT_1204.TOTAL','S-ZORB.FC_1202.TOTAL','S-ZORB.FT_9102.TOTAL','S-ZORB.FT_1001.TOTAL',
        'S-ZORB.TE_2603.DACA','S-ZORB.PDT_1003.DACA','S-ZORB.TE_7106.DACA','S-ZORB.SIS_TEX_3103B.PV','S-ZORB.TC_3102.DACA']

# col3是互信息法出来的前20个特征
col3 = []
for i in [0,1,2,4,5,6,7,9,10,11,12,13,18,19,21,23,25,26,32,38]:
    col3.append(col2[i])

combined_col = col1.copy()
combined_col.extend(col3)
combined_col = list(set(combined_col))
combined_col.append('辛烷值RON.1')
# df.loc[:,combined_col].to_excel("最优子集data.xlsx")

selected_col = ['辛烷值RON','饱和烃,v%（烷烃+环烷烃）','S-ZORB.FC_5001.DACA','S-ZORB.TE_1105.PV','S-ZORB.SIS_FT_3202.PV','S-ZORB.TE_1101.DACA.PV',
                'S-ZORB.TC_2801.PV','S-ZORB.FT_3201.DACA','S-ZORB.AT_1001.DACA','S-ZORB.FT_1004.PV','S-ZORB.FC_1203.PV',
                'S-ZORB.SIS_TE_2802','S-ZORB.PDT_1002.DACA','S-ZORB.FT_5104.PV','S-ZORB.FT_9202.PV','S-ZORB.PT_7103B.DACA',
                '硫含量,μg/g','烯烃,v%','密度(20℃),\nkg/m³','S-ZORB.LT_9101.DACA','S-ZORB.LC_5101.PV']
selected_col.append('辛烷值RON.1')
df_sorted = df.loc[:,selected_col]
df_std = (df_sorted-df_sorted.min())/(df_sorted.max()-df_sorted.min())
# df_std.to_excel("q2df_std.xlsx")
