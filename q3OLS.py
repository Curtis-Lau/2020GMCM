import pandas as pd
import statsmodels.api as sm

df = pd.read_excel('标准数据.xlsx',index_col='样本编号')
selected_col = ['辛烷值RON.1','硫含量,μg/g','饱和烃,v%（烷烃+环烷烃）','烯烃,v%','S-ZORB.LC_5101.PV',
                'S-ZORB.FT_9202.PV','S-ZORB.FT_1004.PV','S-ZORB.FC_1203.PV','S-ZORB.TC_2801.PV',
                'S-ZORB.SIS_TE_2802','S-ZORB.FT_3201.DACA','S-ZORB.LT_9101.DACA','S-ZORB.PDT_1002.DACA',
                'S-ZORB.PT_7103B.DACA','S-ZORB.SIS_FT_3202.PV','S-ZORB.FC_5001.DACA','S-ZORB.TE_1101.DACA.PV',
                '硫含量,μg/g.1'
                ]
df = df.loc[:,selected_col]
df_new = df.drop(['烯烃,v%','S-ZORB.FT_1004.PV','S-ZORB.TC_2801.PV','S-ZORB.SIS_TE_2802'],axis=1)
DataArray = df_new.values
Y = DataArray[:, 0]
X = DataArray[:, 1:-1]

x = sm.add_constant(X)

model = sm.OLS(Y, x).fit()
print(model.summary())

# df_new2 = df.drop(['辛烷值RON.1','S-ZORB.TE_1105.PV','S-ZORB.AT_1001.DACA','密度(20℃),\nkg/m³'],axis=1)
df_new2 = df_new.loc[:,['饱和烃,v%（烷烃+环烷烃）','S-ZORB.FT_3201.DACA','S-ZORB.PT_7103B.DACA','硫含量,μg/g.1']]
DataArray2 = df_new2.values
Y2 = DataArray2[:, -1]
X2 = DataArray2[:, :-1]

x2 = sm.add_constant(X2)

model2 = sm.OLS(Y2, x2).fit()
# print(model2.summary())

# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.preprocessing import StandardScaler
# scaler=StandardScaler()
# #使用不同的正则化系数对模型进行交叉验证
# from sklearn.linear_model import RidgeCV
# alphas = np.logspace(-2,3,100,base=100) #自动生成的一个在1/100到1000之间的数组
# rcv = RidgeCV(alphas=alphas,store_cv_values=True) #将生成的数组传入RidgeCV中返回交叉验证模型的结果
# rcv.fit(X2,Y2) #训练模型
# print('The best alpha is {}'.format(rcv.alpha_)) #最优参数
# print('The r-square is {}'.format(rcv.score(X2,Y2))) #最优R方统计量
#
# #可视化正则化系数与均方误差
# cv_values=rcv.cv_values_
# n_fold,n_alphas=cv_values.shape
# cv_mean=cv_values.mean(axis=0)
# cv_std=cv_values.std(axis=0)
# ub=cv_mean+cv_std/np.sqrt(n_fold)
# lb=cv_mean-cv_std/np.sqrt(n_fold)
# plt.figure(1)
# plt.semilogx(alphas,cv_mean,label='mean_score')
# plt.fill_between(alphas,lb,ub,alpha=0.2)
# plt.xlabel("$\\alpha$")
# plt.ylabel("mean squared errors")
# plt.legend(loc="best")
# # plt.savefig('q3mean_score2.jpg')
# plt.show()
#
# from sklearn.linear_model import Ridge
# ridge=Ridge()
# coefs=[]
# for alpha in alphas:
#     ridge.set_params(alpha=alpha)
#     ridge.fit(X2,Y2)
#     coefs.append(ridge.coef_)
# #然后绘制变量系数随正则化系数变化的轨迹
# plt.figure(2)
# ax=plt.gca()
# ax.plot(alphas,coefs)
# ax.set_xscale('log')
# plt.xlabel('alpha')
# plt.ylabel('weights')
# plt.title('Ridge coefficients as a function of the regularization')
# plt.axis('tight')
# # plt.savefig('q3coef-regulation2.jpg')
# plt.show()