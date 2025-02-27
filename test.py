import datetime
import random

import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
import json
import pandas as pd
import os

from sklearn.neighbors import KernelDensity

import pandas as pd
import os
from matplotlib import rcParams


def append_data_to_excel(file_name, new_data, sheet_name='Sheet1'):
    """
    将新数据追加到指定的 Excel 文件中。

    参数:
    file_name (str): Excel 文件的名称（包括路径，如果不在当前目录）。
    new_data (list): 要追加的新数据，应该是一个列表，其长度与列数相匹配。
    sheet_name (str, 可选): 要写入的 Excel 工作表的名称。默认为 'Sheet1'。
    """
    # 将新数据转换为 DataFrame
    df = pd.DataFrame([new_data], columns=[f'Column{i + 1}' for i in range(len(new_data))])

    # 尝试读取现有的 Excel 文件和工作表
    try:
        # 使用 ExcelWriter 以追加模式打开文件，并指定引擎为 openpyxl
        with pd.ExcelWriter(file_name, engine='openpyxl', mode='a', if_sheet_exists='overlay') as writer:
            # 读取现有的数据（仅读取指定的工作表）
            existing_df = pd.read_excel(file_name, sheet_name=sheet_name, engine='openpyxl')

            # 将新数据追加到现有数据（忽略索引）
            combined_df = pd.concat([existing_df, df], ignore_index=True)

            # 将合并后的数据写回到指定的工作表
            combined_df.to_excel(writer, sheet_name=sheet_name, index=False)

    except FileNotFoundError:
        # 如果文件不存在，直接写入新数据到新的工作表
        df.to_excel(file_name, sheet_name=sheet_name, index=False, engine='openpyxl')


def save_boundary_plot(points_2d, max_y_point, hull_points, filename_prefix='plots/boundary_plot'):
    """
    保存点的边界展示图。

    :param points_2d: 原始点的二维数组。
    :param max_y_point: 角点的坐标。
    :param hull_points: 边界点的坐标。
    :param filename_prefix: 文件名前缀，默认为 'boundary_plot'。
    """
    plt.plot(points_2d[:, 0], points_2d[:, 1], 'o', label='原始点')
    plt.plot(max_y_point[0], max_y_point[1], 'X', label='角点')
    plt.plot(hull_points[:, 0], hull_points[:, 1], 'r-', label='边界')
    plt.fill(hull_points[:, 0], hull_points[:, 1], 'r', alpha=0.2)  # 填充边界区域
    plt.legend()
    plt.xlabel('第一列 (X轴)')
    plt.ylabel('第三列 (Z轴)')
    plt.title('点的边界展示 (只使用第一列和第三列)')

    # 生成唯一文件名
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"{filename_prefix}_{timestamp}.png"

    # 保存图像到文件
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.clf()  # 清理当前图形


def findlefttoppoint():
    # 从 JSON 文件中读取数据
    try:
        with open('newData.json', 'r') as f:
            data = json.load(f)

        # print('读取的数据:', data)

    except FileNotFoundError:
        print('错误: data.json 文件未找到。')
    except json.JSONDecodeError:
        print('错误: JSON 解码失败。请检查文件内容。')
    except Exception as e:
        print(f'发生错误: {e}')
    # print(data)
    points_3d = np.array(data)

    # 只提取第一列和第三列
    points_2d = points_3d[:, [0, 2]]

    # 计算凸包
    # 凸包就是包含点集的最小凸多边形
    hull = ConvexHull(points_2d)

    # 提取边界点
    hull_points = points_2d[hull.vertices]
    # print(hull_points)
    # print("hull_points shape:", hull_points.shape)
    # 区域
    step = 0.05
    min_x_index = np.argmin(hull_points[:, 0])
    min_x = hull_points[min_x_index, 0]
    # print(f'最小的x点坐标{min_x}')
    # filtered_points = hull_points[(hull_points[:, 0] > min_x) & (hull_points[:,0] < min_x + step)]
    filtered_points = hull_points[(hull_points[:, 0] >= min_x) & (hull_points[:, 0] <= min_x + step)]

    # print(hull_points[:,0])

    max_y_index = np.argmax(filtered_points[:, 1])  # 获取 y 最大值的索引
    max_y_point = filtered_points[max_y_index]  # 获取 y 最大值的点
    # print(f'最大的y点坐标{max_y_point}')

    # 求出最大的那个点（x,y,z）
    leftTopPoint3d = points_3d[(points_3d[:, 0] == max_y_point[0]) & (points_3d[:, 2] == max_y_point[1])]
    print(f'dataFromTest{leftTopPoint3d[0]}')
    append_data_to_excel('test.xlsx', leftTopPoint3d[0])
    return np.asarray(leftTopPoint3d)


def compute_kde_density(points, bandwidth=None):
    if bandwidth is None:
        # 自动选择带宽（可以使用 Scott's or Silverman's method）
        bandwidth = np.std(points) * len(points) ** (-1 / 5.)
    kde = KernelDensity(kernel='gaussian', bandwidth=bandwidth).fit(points)
    log_density = kde.score_samples(points)
    density = np.exp(log_density)
    return density


if __name__ == '__main__':
    # ---------------------------------------读取点云--------------------------------------
    # pcd = o3d.io.read_point_cloud("data//投影点测试.pcd")
    # # 如果点云不包含颜色信息，则将点云渲染成灰色
    # if pcd.has_colors == -1:
    #     pcd.paint_uniform_color([0.5, 0.5, 0.5])  # 把所有点渲染为灰色
    # # 将点云的某一个纬度设置为0，在哪个纬度做圆柱邻域搜索，就把对应的纬度设置为0
    # points = np.asarray(pcd.points)
    # xi = points[:, 0]
    # yi = points[:, 1]
    # zi = points[:, 2] - points[:, 2]  # 这里在Z方向上做圆柱邻域搜索
    # project_points = np.c_[xi, yi, zi]
    # # 这行代码创建了一个新的点云对象 project_cloud。在 Open3D 中，PointCloud 是一个用于存储和处理点云数据的类。
    # project_cloud = o3d.geometry.PointCloud()  # 使用numpy生成点云
    # # 这行代码将之前生成的 project_points（包含投影后的点的坐标）转换为 Open3D 可用的格式，并赋值给 project_cloud 的 points 属性。
    # # o3d.utility.Vector3dVector 是 Open3D 提供的一个工具，用于将 Numpy 数组转换为 Open3D 的点云数据结构。
    # project_cloud.points = o3d.utility.Vector3dVector(project_points)
    #
    # # --------------------------------------KDtree搜索--------------------------------------
    # pcd_tree = o3d.geometry.KDTreeFlann(project_cloud)  # 建立KD树索引
    # # ---------------------------------------半径搜索---------------------------------------
    # pcd.colors[15] = [1, 0, 0]  # 给定查询点并渲染为红色
    # [k1, idx1, _] = pcd_tree.search_radius_vector_3d(project_cloud.points[150], 0.5)  # 半径搜索
    # np.asarray(pcd.colors)[idx1[1:], :] = [1, 0, 0]  # 半径搜索结果并渲染为红色
    #
    # o3d.visualization.draw_geometries([pcd])

    # # 假设 corner_points 是你的点云数据
    # corner_points = np.array([[1, 3], [2, 2], [0, 4], [3, 1], [1, 1]])
    #
    # # 计算均值和标准差
    # mean_x = np.mean(corner_points[:, 0])
    # std_x = np.std(corner_points[:, 0])
    # mean_y = np.mean(corner_points[:, 1])
    # std_y = np.std(corner_points[:, 1])
    #
    # # 动态设定权重
    # w_x = 1 + (std_x / mean_x) if mean_x != 0 else 1
    # w_y = 1 + (std_y / mean_y) if mean_y != 0 else 1
    #
    # # 归一化权重
    # total_weight = w_x + w_y
    # w_x_normalized = w_x / total_weight
    # w_y_normalized = w_y / total_weight
    #
    # # 计算得分
    # scores = w_y_normalized * corner_points[:, 1] + w_x_normalized * corner_points[:, 0]
    # best_index = np.argmin(scores)
    # best_point = corner_points[best_index]
    #
    # print("动态权重:", (w_x_normalized, w_y_normalized))
    # print("最靠左上的点坐标:", best_point)
    # 方法二
    # import numpy as np
    # import pandas as pd
    # from sklearn.model_selection import train_test_split
    # from sklearn.ensemble import RandomForestRegressor
    #
    # # 假设 corner_points 是你的点云数据
    # corner_points = np.array([[1, 3], [2, 2], [0, 4], [3, 1], [1, 1]])
    # target = np.array([0, 1, 0, 1, 0])  # 1 表示目标点，0 表示非目标点
    #
    # # 特征构造
    # feature_data = pd.DataFrame(corner_points, columns=['X', 'Y'])
    # feature_data['ratio'] = feature_data['Y'] / (feature_data['X'] + 1)
    #
    # # 数据划分
    # X_train, X_test, y_train, y_test = train_test_split(feature_data, target, test_size=0.2)
    #
    # # 训练模型
    # model = RandomForestRegressor()
    # model.fit(X_train, y_train)
    #
    # # 假设 new_corner_points 是新的点云数据
    # new_corner_points = np.array([[1, 2], [0, 5], [2, 1], [1, 4]])
    #
    # # 特征构造
    # new_feature_data = pd.DataFrame(new_corner_points, columns=['X', 'Y'])
    # new_feature_data['ratio'] = new_feature_data['Y'] / (new_feature_data['X'] + 1)
    #
    # # 进行预测
    # predictions = model.predict(new_feature_data)
    #
    # # 筛选候选点
    # threshold = 0.5
    # candidate_points = new_corner_points[predictions > threshold]
    #
    # if candidate_points.size > 0:
    #     # 找到最靠左上的点
    #     candidate_df = pd.DataFrame(candidate_points, columns=['X', 'Y'])
    #     leftmost_point = candidate_df.loc[candidate_df['X'].idxmin()]
    #     best_point = candidate_df[candidate_df['X'] == leftmost_point['X']].loc[candidate_df['Y'].idxmax()]
    #
    #     print("最靠左上的点坐标:", best_point.values)
    # else:
    #     print("没有找到符合条件的点。")

    # 从 JSON 文件中读取数据
    try:
        with open('newdata.json', 'r') as f:
            data = json.load(f)

        # print('读取的数据:', data)

    except FileNotFoundError:
        print('错误: data.json 文件未找到。')
    except json.JSONDecodeError:
        print('错误: JSON 解码失败。请检查文件内容。')
    except Exception as e:
        print(f'发生错误: {e}')
    points_3d = np.array(data)
    print('读取的数据:', data)

    # 示例三维点数据
    # 随机生成400个三维点
    # np.random.seed(0)  # 设置随机种子以便重现
    # points_3d = np.random.rand(400, 3) * 10  # 生成范围在0到10之间的点

    # 只提取第一列和第三列
    points_2d = points_3d[:, [0, 2]]
    #
    # # 计算凸包
    # # 凸包就是包含点集的最小凸多边形
    hull = ConvexHull(points_2d)
    #
    # 提取边界点
    hull_points = points_2d[hull.vertices]
    # print(hull_points)
    # print("hull_points shape:", hull_points.shape)
    # 区域
    step = 0.01
    min_x_index = np.argmin(hull_points[:, 0])
    min_x = hull_points[min_x_index, 0]
    print(f'最小的x点坐标{min_x}')
    # filtered_points = hull_points[(hull_points[:, 0] > min_x) & (hull_points[:,0] < min_x + step)]
    filtered_points = hull_points[(hull_points[:, 0] >= min_x) & (hull_points[:, 0] <= min_x + step)]
    print(f'过滤后的点{filtered_points}')

    # print(hull_points[:,0])

    max_y_index = np.argmax(filtered_points[:, 1])  # 获取 y 最大值的索引
    max_y_point = filtered_points[max_y_index]  # 获取 y 最大值的点
    print(f'最大的y点坐标{max_y_point}')

    # 求出最大的那个点（x,y,z）
    leftTopPoint3d = points_3d[(points_3d[:, 0] == max_y_point[0]) & (points_3d[:, 2] == max_y_point[1])]
    print('角点坐标', leftTopPoint3d.shape)
    # 绘制点和边界
    rcParams['font.family'] = 'SimHei'
    plt.plot(points_2d[:, 0], points_2d[:, 1], 'o', label='原始点')
    plt.plot(max_y_point[0], max_y_point[1], 'X', label='角点')
    plt.plot(hull_points[:, 0], hull_points[:, 1], 'r-', label='边界')
    plt.fill(hull_points[:, 0], hull_points[:, 1], 'r', alpha=0.2)  # 填充边界区域
    plt.legend()
    plt.xlabel('第一列 (X轴)')
    plt.ylabel('第三列 (Z轴)')
    plt.title('点的边界展示 (只使用第一列和第三列)')
    plt.show()
