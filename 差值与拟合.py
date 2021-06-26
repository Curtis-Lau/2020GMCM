import numpy as np
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
from scipy import interpolate as spi

## 一维插值
x = np.linspace(0,10,11)
y = np.sin(x)
x_ = np.linspace(0,10,101)

# 一阶样条(线性插值)
ipo1 = spi.splrep(x,y,k=1)
y_1 = spi.splev(x_,ipo1)

# 三阶样条
ipo3 = spi.splrep(x,y,k=3)
y_3 = spi.splev(x_,ipo3)

# plt.figure(figsize=(15,12))
# plt.scatter(x,y)
# plt.plot(x_,y_1,'o',label='线性插值',color='r',linestyle='dashed')
# plt.plot(x_,y_3,'p',label='三阶样条',color='b',linestyle='-.')
# plt.legend()
# plt.xlabel("x")
# plt.ylabel("y")
# plt.show()

# ## 也可以调用interp1d
# for k in ["nearest","zero","slinear","quadratic","cubic"]:
#     # "nearest","zero"为阶梯插值
#     # slinear 线性插值
#     # "quadratic","cubic" 为2阶、3阶B样条曲线插值
#     f = spi.interp1d(x,y,kind=k,)
#     ynew = f(x_)
#     plt.plot(x_, ynew, label=str(k))
# plt.legend()
# plt.show()

# 拉格朗日插值
def lagrange(x, y, num_points, x_test):
    # 所有的基函数值，每个元素代表一个基函数的值
    l = np.zeros(shape=(num_points, ))
    # 计算第k个基函数的值
    for k in range(num_points):
        # 乘法时必须先有一个值
        # 由于l[k]肯定会被至少乘n次，所以可以取1
        l[k] = 1
        # 计算第k个基函数中第k_个项（每一项：分子除以分母）
        for k_ in range(num_points):
            # 这里没搞清楚，书中公式上没有对k=k_时，即分母为0进行说明
            # 有些资料上显示k是不等于k_的
            if k != k_:
                # 基函数需要通过连乘得到
                l[k] = l[k]*(x_test-x[k_])/(x[k]-x[k_])
            else:
                pass
    # 计算当前需要预测的x_test对应的y_test值
    L = 0
    for i in range(num_points):
        # 求所有基函数值的和
        L += y[i]*l[i]
    return L

# y_predict = [lagrange(x,y,len(x),x_i) for x_i in x_]
# plt.plot(x_,y_predict,label='lagrange')
# plt.legend()
# plt.show()

## 二维插值
x = np.linspace(-1, 1, 20)
y = np.linspace(-1,1,20)
x, y = np.meshgrid(x, y)   #20*20的网格数据

def func(x, y):
    return (x+y)*np.exp(-5.0*(x**2 + y**2))
z = func(x,y) # 计算每个网格点上的函数值

# fig = plt.figure(figsize=(9, 6))

# 原始数据图
# ax = plt.subplot(121,projection = '3d')
# surf = ax.plot_surface(x,y,z, rstride=2, cstride=2, cmap = cm.coolwarm,linewidth=0.5, antialiased=True)
# ax.set_xlabel('x')
# ax.set_ylabel('y')
# ax.set_zlabel('f(x, y)')
# plt.colorbar(surf,shrink=0.5,aspect=5) #标注

# 二维3阶样条插值图
# newfunc = spi.interp2d(x, y, z, kind='cubic')

# 计算100*100的网格上的插值
# xnew = np.linspace(-1,1,100)#x
# ynew = np.linspace(-1,1,100)#y
# fnew = newfunc(xnew, ynew)#仅仅是y值   100*100的值  np.shape(fnew) is 100*100
# mesh形成平面
# xnew, ynew = np.meshgrid(xnew, ynew)
# 画图
# ax2 = plt.subplot(122,projection = '3d')
# surf2 = ax2.plot_surface(xnew, ynew, fnew, rstride=2, cstride=2, cmap=plt.get_cmap('rainbow'))
# 设置轴标签
# ax2.set_xlabel('xnew')
# ax2.set_ylabel('ynew')
# ax2.set_zlabel('fnew(x, y)')
# 画等高线
# ax2.contour(xnew,ynew,fnew,zdir='z',offset=-0.3,cmap=plt.get_cmap('rainbow'))
# colorbar标注
# plt.colorbar(surf2, shrink=0.5, aspect=5)

# plt.show()

## 拟合
from scipy.optimize import leastsq

# X = np.array([8.19,2.72,6.39,8.71,4.7,2.66,3.78])
# Y = np.array([7.01,2.78,6.47,6.71,4.1,4.23,4.05])
###需要拟合的函数func及误差error###
# def func(p,x):
#     k,b=p
#     return k*x+b
#
# def error(p,x,y):
#     return func(p,x)-y #x、y都是列表，故返回值也是个列表

# p0 = [1,20]
# param = leastsq(error,p0,args=(X,Y))[0]
# k = param[0]
# b = param[1]
# fitted_y = k*X+b

# plt.scatter(X,Y,label='样本数据')
# plt.plot(X,fitted_y,label='拟合曲线')
# plt.legend()
# plt.show()

## 多项式拟合
# x = [1,2,3,4,5,6,7,8]
# y = [1,4,9,13,30,25,49,70]
# func_ = np.polyfit(x,y,3)           # 用2次多项式拟合x，y数组
# y_fitted = np.polyval(func_,x)       # 拟合完之后用这个函数来生成多项式并代入数据得到拟合值
# plt.scatter(x,y,marker='o',label='original datas')#对原始数据画散点图
# plt.plot(x,y_fitted,ls='--',c='red',label='fitting with second-degree polynomial')#对拟合之后的数据，也就是x，c数组画图
# plt.legend()
# plt.show()