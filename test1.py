import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
import json

from sklearn.neighbors import KernelDensity


def findlefttoppoint():
    # 从 JSON 文件中读取数据
    try:
        with open('data.json', 'r') as f:
            data = json.load(f)

        # print('读取的数据:', data)

    except FileNotFoundError:
        print('错误: data.json 文件未找到。')
    except json.JSONDecodeError:
        print('错误: JSON 解码失败。请检查文件内容。')
    except Exception as e:
        print(f'发生错误: {e}')
    points_3d = np.array(data)

    # 示例三维点数据
    # 随机生成400个三维点
    # np.random.seed(0)  # 设置随机种子以便重现
    # points_3d = np.random.rand(400, 3) * 10  # 生成范围在0到10之间的点

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
    step = 0.1
    min_x_index = np.argmin(hull_points[:, 0])
    min_x = hull_points[min_x_index, 0]
    # print(f'最小的x点坐标{min_x}')
    # filtered_points = hull_points[(hull_points[:, 0] > min_x) & (hull_points[:,0] < min_x + step)]
    filtered_points = hull_points[(hull_points[:, 0] >= min_x) & (hull_points[:, 0] <= min_x + step)]
    # print(f'过滤后的点{filtered_points}')

    # print(hull_points[:,0])

    max_y_index = np.argmax(filtered_points[:, 1])  # 获取 y 最大值的索引
    max_y_point = filtered_points[max_y_index]  # 获取 y 最大值的点
    # print(f'最大的y点坐标{max_y_point}')

    # 求出最大的那个点（x,y,z）
    leftTopPoint3d = points_3d[(points_3d[:, 0] == max_y_point[0]) & (points_3d[:, 2] == max_y_point[1])]
    # print('角点坐标', leftTopPoint3d)
    print(leftTopPoint3d)
    return np.asarray(leftTopPoint3d)
    # 绘制点和边界
    # plt.plot(points_2d[:, 0], points_2d[:, 1], 'o', label='原始点')
    # plt.plot(max_y_point[0], max_y_point[1], 'X', label='角点')
    # plt.plot(hull_points[:, 0], hull_points[:, 1], 'r-', label='边界')
    # plt.fill(hull_points[:, 0], hull_points[:, 1], 'r', alpha=0.2)  # 填充边界区域
    # plt.legend()
    # plt.xlabel('第一列 (X轴)')
    # plt.ylabel('第三列 (Z轴)')
    # plt.title('点的边界展示 (只使用第一列和第三列)')
    # plt.show()

def compute_kde_density(points,bandwidth=None):
    if bandwidth is None:
        # 自动选择带宽（可以使用 Scott's or Silverman's method）
        bandwidth = np.std(points) * len(points) ** (-1 / 5.)
    kde = KernelDensity(kernel='gaussian', bandwidth=bandwidth).fit(points)
    log_density = kde.score_samples(points)
    density = np.exp(log_density)
    return density

if __name__ == '__main__':
    findlefttoppoint()

