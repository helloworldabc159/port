import matplotlib.pyplot as plt
import pandas as pd
from scipy import signal
path = r'C:\Users\wxw\Desktop\1207LSM_FIR.xlsx'
data = pd.DataFrame(pd.read_excel(path))  # 读取数据,设置None可以生成一个字典，字典中的key值即为sheet名字，此时不用使用DataFram，会报错
yaw_SVD_FIR = data.iloc[:, 1].values   #处理后的偏航角
pitch_SVD_FIR = data.iloc[:, 2].values #处理后的俯仰角
yaw_gt = data.iloc[:, 10].values  #实际偏航角
pitch_gt = data.iloc[:, 9].values  #实际俯仰角
plt.figure()
plt.ylim((89, 90.8))  #设置y轴的范围
plt.plot(yaw_gt, color='red')  #绘制实际偏航角（红色）
plt.plot(yaw_SVD_FIR, color='blue') #绘制处理后的偏航角（蓝色）

plt.figure()
plt.ylim((0, 3))     #与上面几行代码同理
plt.plot(pitch_gt, color='red')
plt.plot(pitch_SVD_FIR, color='blue')

plt.show()


exit(0)

path1 = r'C:\Users\wxw\Desktop\HHG_Queue提取后.xlsx'
path2 = r'C:\Users\wxw\Desktop\RANSAC_SVD_5m.xlsx'
data1 = pd.DataFrame(pd.read_excel(path1))  # 读取数据,设置None可以生成一个字典，字典中的key值即为sheet名字，此时不用使用DataFram，会报错
data2 = pd.DataFrame(pd.read_excel(path2))  # 读取数据,设置None可以生成一个字典，字典中的key值即为sheet名字，此时不用使用DataFram，会报错

result1 = data1.iloc[:, 1].values  # 获取列明为院系，内容为动力的内容
result2 = data2.iloc[:, 4].values  # 获取列明为院系，内容为动力的内容
# print(result1)
# print(result2)



data = [0.5, 1.7, 1.9, 2.3, 3.4, 3.5, 3.9, 4.2, 5.4, 5.7, 6.1, 8.2]
data = result2   #result2是要处理的数据
# 创建一个3阶的低通巴特沃斯滤波器，截止频率为0.1
# b 是滤波器的分子系数，a 是分母系数。
b, a = signal.butter(3, 0.1)
print(b, a)
# 应用滤波器
ss = signal.filtfilt(b, a, data)
print(ss)

# 手动滤波
output = []
for i in range(len(data)):
    if i == 0:
        y = b[0] * data[i] + 0
    if i == 1:
        y = b[0] * data[i] + b[1] * data[i - 1] - a[1] * output[i - 1]
    if i == 2:
        y = b[0] * data[i] + b[1] * data[i - 1] + b[2] * data[i - 2] - a[1] * output[i - 1] - a[2] * output[i - 2]
    if i > 2:
        y = b[0] * data[i] + b[1] * data[i - 1] + b[2] * data[i - 2] + b[3] * data[i - 3] \
            - a[1] * output[i - 1] - a[2] * output[i - 2] - a[3] * output[i - 3]
    output.append(y)
print(output)
plt.ylim((89.2, 90.8))
plt.plot(data[:], color='red')
plt.plot(output[:], color='green')
# plt.plot(ss, color='blue')
plt.show()
