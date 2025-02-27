import open3d as o3d
import numpy as np

pcd = o3d.io.read_point_cloud("yolo_color.pcd")
plane_model, inliers = pcd.segment_plane(distance_threshold=0.005,
                                         ransac_n=10,
                                         num_iterations=1000)
[a, b, c, d] = plane_model
print(f"Plane equation: {a:.6f}x + {b:.6f}y + {c:.6f}z + {d:.6f} = 0")

inlier_cloud = pcd.select_by_index(inliers)
inlier_cloud.paint_uniform_color([1.0, 0, 0])
outlier_cloud = pcd.select_by_index(inliers, invert=True)
outlier_cloud.paint_uniform_color([0, 1, 0])
o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud])


def MSAC(data):
    number = data.shape[1]  # 点的总数
    iter = 5000  # 迭代次数
    sigma = 0.005  # 阈值
    preF = np.inf  # 初始最小代价函数值

    bestplane = None

    for _ in range(iter):
        idx = np.random.choice(number, 3, replace=False)  # 随机选择3个点的索引
        sample = data[:, idx]  # 获取这3个点的数据

        p1 = sample[:, 0]
        p2 = sample[:, 1]
        p3 = sample[:, 2]

        v1 = p2 - p1
        v2 = p3 - p1

        # 检查点是否共线
        if np.linalg.norm(np.cross(v1, v2)) < 1e-6:
            continue  # 点共线，跳过此迭代

        nv = np.cross(v1, v2)  # 计算法向量
        nv = nv / np.linalg.norm(nv)  # 规范化法向量

        nv = np.append(nv, -np.dot(nv, p1))  # 计算平面方程的d参数

        if nv[0] < 0 and abs(nv[0]) > 0.01:
            nv = -nv  # 确保法向量的第一个分量为正

        # 计算所有点到平面的距离
        mask = np.abs(np.dot(nv, np.vstack((data, np.ones((1, data.shape[1]))))))

        F1 = np.sum(mask[mask < sigma])  # 累加所有距离小于sigma的点的距离
        F2 = np.sum(mask > sigma) * 0.01  # 距离大于sigma的点的数量乘以惩罚因子
        F = F1 + F2  # 总代价函数值

        if F < preF:
            preF = F  # 更新最小代价函数值
            bestplane = nv  # 更新最佳平面

    mask = np.abs(np.dot(bestplane, np.vstack((data, np.ones((1, data.shape[1])))))) < sigma
    inliers = data[:, mask]  # 获取内点

    para = {
        'a': bestplane[0],
        'b': bestplane[1],
        'c': bestplane[2],
        'd': bestplane[3],
    }

    return para, inliers
pcd = o3d.io.read_point_cloud("yolo_color.pcd")
points = np.asarray(pcd.points).T

tail_params, tail_inliers = MSAC(points)
# 将内点转换回点云
inlier_pcd = o3d.geometry.PointCloud()
inlier_pcd.points = o3d.utility.Vector3dVector(tail_inliers.T)
# 打印平面方程参数
print("船尾面平面方程参数:")
print(f"a: {tail_params['a']}, b: {tail_params['b']}, c: {tail_params['c']}, d: {tail_params['d']}")
# 绘制结果
o3d.visualization.draw_geometries([inlier_pcd])
