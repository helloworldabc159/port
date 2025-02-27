import json
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 假设你有一个 JSON 文件 'data.json'
with open('newdata.json', 'r') as f:
    data = json.load(f)

# 检查数据结构
print(data)  # 打印数据以确认格式

# 提取点的坐标
x = []
y = []
z = []

# 确保 data 是一个列表
if isinstance(data, list):
    for i in range(0, len(data), 3):  # 每三个元素为一组 (x, y, z)
        if i + 2 < len(data):  # 确保有足够的元素
            x.append(data[i])     # x 坐标
            y.append(data[i + 1]) # y 坐标
            z.append(data[i + 2]) # z 坐标

# 创建三维图
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# 绘制点
ax.scatter(x, y, z)

# 设置标签
ax.set_xlabel('X轴')
ax.set_ylabel('Y轴')
ax.set_zlabel('Z轴')

# 显示图形
plt.show()
