import open3d as o3d
import cv2
import numpy as np
import pandas as pd

# while True:
# 读取点云数据
# point_cloud = o3d.io.read_point_cloud("8.pcd")
data = pd.read_csv(r"/home/g/take/20231108124928RS.csv", encoding='utf-8')  # 读取csv文件
data_234 = data.iloc[:, 0:3]  # 这里做的是切割，因为这里只用到了其收集数据的第1列到第3列，即x,y,z 可根据自身需要设置 iloc:左到右不到
Data1 = np.array(data_234)  # 转换为numpy方便计算
print(Data1)
point_cloud = o3d.geometry.PointCloud()
point_cloud.points = o3d.utility.Vector3dVector(Data1)


# 读取图像
image = cv2.imread(r"/home/g/take/20231108124924.jpg")

# 固定参数：相机内参、旋转向量、平移向量
camera_matrix = np.array([[3.4337542953755502e+02, 0., 3.4591435545841017e+02],
                           [0, 3.4023110616706424e+02, 2.3949073697355124e+02],
                           [0, 0, 1]])  # 内参矩阵
rvec = np.float64([1.19003915, -1.1942513, 1.17190771])  # 旋转向量
tvec = np.float64([-0.06198415, -0.04377533, -0.09804805])  # 平移向量

# 将旋转向量转换为旋转矩阵
rotation_matrix, _ = cv2.Rodrigues(rvec)

# 将点云坐标转换到相机坐标系
point_cloud_transformed = np.asarray(point_cloud.points).T

# 使用旋转矩阵和平移矩阵将点云从世界坐标系变换到相机坐标系
transformed_point_cloud = np.dot(rotation_matrix, point_cloud_transformed) + tvec.reshape(3, 1)

image_points = np.dot(camera_matrix, transformed_point_cloud)
image_points /= image_points[2, :]  # 归一化

# 将图像坐标转换为整数
image_points = np.round(image_points[:2, :]).astype(int)

# 创建新的点云对象，带有颜色信息
colored_point_cloud = o3d.geometry.PointCloud()
colored_point_cloud.points = o3d.utility.Vector3dVector(point_cloud_transformed.T)

# 进行颜色校正
colored_point_cloud_colors = []

for i in range(image_points.shape[1]):
    u, v = image_points[:, i]
    if 0 <= u < image.shape[1] and 0 <= v < image.shape[0]:
        current_color = image[v, u][::-1]  # 获取图像上的颜色并反转BGR为RGB
        colored_point_cloud_colors.append(current_color / 255.0)  # 归一化颜色
    else:
        colored_point_cloud_colors.append([0, 0, 0])

colored_point_cloud.colors = o3d.utility.Vector3dVector(colored_point_cloud_colors)

# 创建Open3D窗口并可视化点云
o3d.visualization.draw_geometries([colored_point_cloud])

# 保存彩色点云到桌面（或其他路径）
#o3d.io.write_point_cloud("C:/Users/YourUsername/Desktop/colored_point_cloud.ply", colored_point_cloud)
