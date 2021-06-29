import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = 'SimHei'
plt.rcParams['axes.unicode_minus'] = False
import pandas as pd

df = pd.read_excel("q4result.xlsx",index_col='样本编号')
df_313 = df.loc[313]

x_4 = [479.884,429.884,379.884,329.884,279.884,250,250]
x_4_ = [0,1,2,3,4,5,6]

x_5 = [10.392,11.392,12.195,12.195]
x_5_ = [0,1,2,3]

x_6 = [706.646,656.646,606.646,556.646,506.646,456.646,406.646,356.646,344.646,344.646]
x_6_ = [0,1,2,3,4,5,6,7,8,9]

def s(x_6):
    return 7.7741-0.0716*55.05+0.0018*x_6+0.1897*2.2197

s_ = [8.5,8.04675,7.5935,7.14025,6.687,6.23375,5.7805,5.32725,4.874,4.874]
_s_ = [0,1,2,3,4,5,6,7,8,9]

x_10 = [271.5438,221.5438,171.5438,150,150]
x_10_ = [0,1,2,3,4]

x_11 = []
j = 1788.629
for i in range(35):
    x_11.append(j)
    j += 50
x_11.extend([3500,3500])
x_11_ = list(range(37))

y = []
j = 89.05
for i in range(35):
    y.append(j)
    j += 0.0226
y.extend([89.84377,89.84377])
y_ = list(range(37))

plt.figure(1)
plt.plot(x_4_,x_4)
plt.xlim(0,6)
plt.xlabel('调整次数')
plt.savefig('x4轨迹图.jpg')
plt.show()

plt.figure(2)
plt.plot(x_5_,x_5)
plt.xlim(0,3)
plt.xlabel('调整次数')
plt.savefig('x5轨迹图.jpg')
plt.show()

plt.figure(3)
plt.plot(x_6_,x_6)
plt.xlabel('调整次数')
plt.savefig('x6轨迹图.jpg')
plt.xlim(0,9)
plt.show()

plt.figure(4)
plt.plot(_s_,s_)
plt.xlim(0,9)
plt.xlabel('调整次数')
plt.savefig('硫含量轨迹图.jpg')
plt.show()

plt.figure(5)
plt.plot(x_10_,x_10)
plt.xlim(0,4)
plt.xlabel('调整次数')
plt.savefig('x10轨迹图.jpg')
plt.show()

plt.figure(6)
plt.plot(x_11_,x_11)
plt.xlim(0,37)
plt.xlabel('调整次数')
plt.savefig('x11轨迹图.jpg')
plt.show()