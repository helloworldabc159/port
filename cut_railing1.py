import numpy as np
import open3d as o3d

from 切割方法.cut_railing import moved_plane_equations

# 有v1标识的注解是为了理解代码写的注释
# 读取点云文件，假设文件名为"point_cloud.ply"或"point_cloud.pcd"

# point_cloud_file = r"E:/post gratuation/anji/imgandpcd/2024-10-17/pcd/20241017/2024_10_17_00_00_20.pcd"  # 替换为你的点云文件路径
# point_cloud_file =  pcd_file_paths
# point_cloud = o3d.io.read_point_cloud(point_cloud_file)
# point_cloud_np = np.asarray(point_cloud.points)  # 转换为numpy数组

point_cloud_np = None

# 船尾点云
pcd_chuanwei = None

# 定义平面B的方程，该平面不会移动
# plane_B_equation = [0.992700, 0.119837, 0.013620, -45.766523]
plane_B_equation = None
thickness_B = 1  # 平面B的厚度

# 已知平面方程A的系数
plane_equation_A = [0, 0, 1, -10]  # 移除了多余的空格
thickness_A = 0.1  # 平面A的厚度


# 移动的步数
steps = 1000


# 添加滤波代码
# points = point_cloud_np
# print(points)
# after_filter = np.logical_and.reduce([
#     points[:, 0] >= 46, points[:, 0] <= 100,  # 水平方向过滤  X
#     points[:, 1] >= -40, points[:, 1] <= -10,    # 纵深  Y
#     # points[:, 2] >= -10, points[:, 2] <= 20, # 垂直方向  Z
# ])
# point_cloud_np_filtered = points[after_filter]
#
# # 更新 point_cloud_np 为过滤后的点云数据
# point_cloud_np = point_cloud_np_filtered

# 创建新的点云对象
# point_cloud2 = o3d.geometry.PointCloud()
# point_cloud2.points = o3d.utility.Vector3dVector(point_cloud_np_filtered)
# ------------------------------------------------------------------------------
# # # 添加滤波代码
# points2 = point_cloud_np
# after_filter2 = np.logical_and.reduce([
#     points2[:, 0] >= 46, points2[:, 0] <= 100,  # 水平方向过滤  X
#     points2[:, 1] >= -40, points2[:, 1] <= 5,    # 纵深  Y
#     # points[:, 2] >= -10, points[:, 2] <= 20, # 垂直方向  Z
# ])
# point_cloud_np_filtered2 = points2[after_filter]
#
# # 更新 point_cloud_np 为过滤后的点云数据
# point_cloud_np2 = point_cloud_np_filtered2
#
# # 创建新的点云对象
# point_cloudK = o3d.geometry.PointCloud()
# point_cloudK.points = o3d.utility.Vector3dVector(point_cloud_np_filtered)
# --------------------------------------------------------------------------------


# v1：判断是否点在面上
def is_point_on_plane(plane_equation, point):
    A, B, C, D = plane_equation
    x, y, z = point
    return np.isclose(A * x + B * y + C * z + D, 0)

# v1:综上所述，这段代码的作用是：如果提供了 exclude_plane 平面方程，那么代码会排除掉
# v1:位于这个平面上的点，只保留那些不在 exclude_plane 上的点。这样，返回的 point_cloud_np 数组
# v1:就只包含那些位于 exclude_plane 平面之外的点。
# v1：当有点不在面上返回true，点全在面上返回false
# 在 detect_points_nearby 函数中修改迭代对象
def detect_points_nearby(plane_equation, point_cloud_np, exclude_plane=None):
    A, B, C, D = plane_equation

    def is_point_above_or_below(point):
        x, y, z = point
        # v1:不在面上返回true，在面上返回false
        return A * x + B * y + C * z + D > 0 or A * x + B * y + C * z + D < 0

    if exclude_plane:
        point_cloud_np = point_cloud_np[~np.apply_along_axis(lambda p: is_point_on_plane(exclude_plane, p), axis=1, arr=point_cloud_np)]

    return any(is_point_above_or_below(point) for point in point_cloud_np)





def move_plane_equation(plane_equation, steps, point_cloud, plane_B_equation):
    A, B, C, D = plane_equation
    new_plane_equations = []
    move_step = C * 0.1  # 定义移动步长为C的0.1倍
    lowest_point = None  # 记录平面A检测到的最低点

    def is_point_nearby(plane, point, exclude_plane, threshold):
        # 计算点到平面A的距离
        distance_to_plane = abs(plane[0] * point[0] + plane[1] * point[1] + plane[2] * point[2] + plane[3]) / np.sqrt(
            plane[0] ** 2 + plane[1] ** 2 + plane[2] ** 2)

        # 计算点到平面B的距离
        distance_to_B_plane = abs(
            exclude_plane[0] * point[0] + exclude_plane[1] * point[1] + exclude_plane[2] * point[2] + exclude_plane[3]
        ) / np.sqrt(exclude_plane[0] ** 2 + exclude_plane[1] ** 2 + exclude_plane[2] ** 2)

        # 确保点在平面A的附近，但不在平面B的1单位距离内
        return distance_to_plane < threshold and distance_to_B_plane > threshold

    while True:
        # 检查平面A下方1单位附近且不在平面B的1单位距离范围内的点云
        points_below_plane = [
            point for point in point_cloud
            if is_point_nearby([A, B, C, D + move_step], point, plane_B_equation, 1)  # 阈值从 0.1 改为 1
        ]

        if points_below_plane:
            # 如果找到点，记录当前平面方程和最低点
            new_plane_equations.append([A, B, C, D])
            D += move_step  # 向下移动平面A
            # 更新最低点坐标
            current_lowest = min(points_below_plane, key=lambda p: p[2])
            if lowest_point is None or current_lowest[2] < lowest_point[2]:
                lowest_point = current_lowest
        else:
            # 如果平面A下方无点云，则停止移动
            print(f"找到合适的平面方程: {A}x + {B}y + {C}z + {D} = 0")
            if lowest_point is not None:
                print(f"平面A检测到的最低点坐标: {lowest_point}")
            break

        # 如果移动次数超过设定的步数，则停止移动
        if len(new_plane_equations) > steps:
            print(f"未找到合适的平面方程，已达到最大移动步数: {steps}")
            break

    return new_plane_equations,lowest_point





# def move_plane_equation(plane_equation, steps, point_cloud, plane_B_equation):
#     A, B, C, D = plane_equation
#     new_plane_equations = []
#     move_step = C * 0.1  # 定义移动步长为C的0.1倍
#     lowest_point = None  # 记录平面A检测到的最低点
#
#     def is_point_nearby(plane, point, exclude_plane, threshold):
#         # 计算点到平面A的距离
#         distance_to_plane = abs(plane[0] * point[0] + plane[1] * point[1] + plane[2] * point[2] + plane[3]) / np.sqrt(
#             plane[0] ** 2 + plane[1] ** 2 + plane[2] ** 2)
#
#         # 计算点到平面B的距离
#         distance_to_B_plane = abs(
#             exclude_plane[0] * point[0] + exclude_plane[1] * point[1] + exclude_plane[2] * point[2] + exclude_plane[3]
#         ) / np.sqrt(exclude_plane[0] ** 2 + exclude_plane[1] ** 2 + exclude_plane[2] ** 2)
#
#         # 确保点在平面A的附近，但不在平面B的1单位距离内
#         return distance_to_plane < threshold and distance_to_B_plane > threshold
#
#     while True:
#         # 检查平面A下方1单位附近且不在平面B的1单位距离范围内的点云
#         points_below_plane = [
#             point for point in point_cloud
#             if is_point_nearby([A, B, C, D + move_step], point, plane_B_equation, 1)  # 阈值从 0.1 改为 1
#         ]
#
#         if points_below_plane:
#             # 如果找到点，记录当前平面方程和最低点
#             new_plane_equations.append([A, B, C, D])
#             D += move_step  # 向下移动平面A
#             # 更新最低点坐标
#             current_lowest = min(points_below_plane, key=lambda p: p[2])
#             if lowest_point is None or current_lowest[2] < lowest_point[2]:
#                 lowest_point = current_lowest
#         else:
#             # 如果平面A下方无点云，则停止移动
#             print(f"找到合适的平面方程: {A}x + {B}y + {C}z + {D} = 0")
#             if lowest_point is not None:
#                 print(f"平面A检测到的最低点坐标: {lowest_point}")
#             break
#
#         # 如果移动次数超过设定的步数，则停止移动
#         if len(new_plane_equations) > steps:
#             print(f"未找到合适的平面方程，已达到最大移动步数: {steps}")
#             break
#
#     return new_plane_equations,lowest_point
#



#
# #
# def move_plane_equation(plane_equation, steps, point_cloud, plane_B_equation):
#     A, B, C, D = plane_equation
#     new_plane_equations = []
#     move_step = C * 0.01  # 定义移动步长为C的0.1倍
#     lowest_point = None  # 记录平面A检测到的最低点
#     found_above = False  # 标记是否检测到上方点云
#
#     def is_point_nearby(plane, point, exclude_plane, lower_threshold, exclude_threshold):
#         # 计算点到当前平面的距离
#         distance_to_plane = abs(plane[0] * point[0] + plane[1] * point[1] + plane[2] * point[2] + plane[3]) / np.sqrt(
#             plane[0] ** 2 + plane[1] ** 2 + plane[2] ** 2
#         )
#         # 计算点到平面B的距离
#         distance_to_B_plane = abs(
#             exclude_plane[0] * point[0] + exclude_plane[1] * point[1] + exclude_plane[2] * point[2] + exclude_plane[3]
#         ) / np.sqrt(exclude_plane[0] ** 2 + exclude_plane[1] ** 2 + exclude_plane[2] ** 2)
#
#         # 检查点是否在下方的阈值范围内，并且在平面B的0.2单位距离之外
#         return (
#             distance_to_plane <=lower_threshold
#             and distance_to_B_plane > exclude_threshold
#         )
#
#     while True:
#         if not found_above:
#             # 检查平面A下方2单位距离内且不在平面B 0.5单位距离内的点云
#             points_above_plane = [
#                 point for point in point_cloud
#                 if is_point_nearby([A, B, C, D - move_step], point, plane_B_equation, 4, 1)
#             ]
#             if points_above_plane:
#                 found_above = True  # 记录已检测到上方点云
#                 print(f"首次检测到上方点云，继续检测下方点云。")
#         else:
#             # 检查平面A下方2单位距离内且不在平面B 0.2单位距离内的点云
#             points_below_plane = [
#                 point for point in point_cloud
#                 if is_point_nearby([A, B, C, D + move_step], point, plane_B_equation, 4 ,1)
#             ]
#
#             if not points_below_plane:
#                 print(f"找到合适的平面方程: {A}x + {B}y + {C}z + {D} = 0")
#                 if lowest_point is not None:
#                     print(f"平面A检测到的最低点坐标: {lowest_point}")
#                 break
#
#             # 如果找到下方点，记录当前平面方程和最低点
#             new_plane_equations.append([A, B, C, D])
#             D += move_step  # 向下移动平面A
#             current_lowest = min(points_below_plane, key=lambda p: p[2])
#             if lowest_point is None or current_lowest[2] < lowest_point[2]:
#                 lowest_point = current_lowest
#
#         # 如果移动次数超过设定的步数，则停止移动
#         if len(new_plane_equations) > steps:
#             print(f"未找到合适的平面方程，已达到最大移动步数: {steps}")
#             break
#
#     return new_plane_equations, lowest_point
#



#
# def visualize_with_open3d(point_cloud_np, plane_equations, plane_B_equation, thickness_A=0.01, thickness_B=0.1):
#     if point_cloud_np.size == 0:
#         print("点云数据为空，无法进行可视化。")
#         return
#
#     # 创建点云对象
#     pcd = o3d.geometry.PointCloud()
#     pcd.points = o3d.utility.Vector3dVector(point_cloud_np)
#     pcd.paint_uniform_color([0, 0, 1])  # 点云颜色：蓝色
#
#     # 创建一个可视化窗口
#     vis = o3d.visualization.Visualizer()
#     vis.create_window(window_name='Point Cloud and Planes', width=800, height=600)
#
#     # 设置视点参数
#     ctr = vis.get_view_control()
#     ctr.set_front([0, 0, -1])
#     ctr.set_up([0, 1, 0])
#     ctr.set_lookat([0, 0, 0])
#
#     # 添加点云
#     vis.add_geometry(pcd)
#
#     # 初始化最低点
#     _,lowest_point = move_plane_equation(plane_equation_A,  steps, point_cloud_np, plane_B_equation)
#
#
#     # 绘制平面A
#     for eq in plane_equations:
#         A, B, C, D = eq
#         # 调整网格范围以匹配点云范围
#         x_range = np.linspace(min(point_cloud_np[:, 0]), max(point_cloud_np[:, 0]), 10)
#         y_range = np.linspace(min(point_cloud_np[:, 1]), max(point_cloud_np[:, 1]), 10)
#         X, Y = np.meshgrid(x_range, y_range)
#         Z = (-A / C * X - B / C * Y - D / C)
#         plane_points = np.vstack((X.flatten(), Y.flatten(), Z.flatten())).T
#         plane_points[:, 2] += thickness_A
#         plane_pcd = o3d.geometry.PointCloud()
#         plane_pcd.points = o3d.utility.Vector3dVector(plane_points)
#         plane_pcd.paint_uniform_color([1, 0, 0])  # 平面A颜色：红色
#         vis.add_geometry(plane_pcd)
#
#
#
#     # 绘制平面B
#     A, B, C, D = plane_B_equation
#     x_range = np.linspace(min(point_cloud_np[:, 0]), max(point_cloud_np[:, 0]), 10)
#     y_range = np.linspace(min(point_cloud_np[:, 1]), max(point_cloud_np[:, 1]), 10)
#     X, Y = np.meshgrid(x_range, y_range)
#     Z = (-A / C * X - B / C * Y - D / C)
#     plane_points_B = np.vstack((X.flatten(), Y.flatten(), Z.flatten())).T
#     plane_points_B[:, 2] += thickness_B
#     plane_pcd_B = o3d.geometry.PointCloud()
#     plane_pcd_B.points = o3d.utility.Vector3dVector(plane_points_B)
#     plane_pcd_B.paint_uniform_color([0, 1, 0])  # 平面B颜色：绿色
#     # vis.add_geometry(plane_pcd_B)
#
#     # 根据最低点生成新点，并可视化由 (0, 0, 0)、最低点和生成点构成的平面
#     if lowest_point is not None:
#         x_lowest, y_lowest, z_lowest = lowest_point
#         y_new = np.random.uniform(0, 1)
#         new_point = np.array([x_lowest, y_new, z_lowest])
#
#         # 打印三个点的坐标
#         print(f"构成黄色平面的三个点坐标：")
#         print(f"原点: (0, 0, 0)")
#         print(f"最低点: {lowest_point}")
#         print(f"新生成点: {new_point}")
#
#         # 构建新平面上的三个点
#         origin = np.array([0, 0, 0])
#         plane_points_new = np.array([origin, lowest_point, new_point])
#
#         # 计算新平面的法向量和平面方程
#         normal_vector = np.cross(lowest_point - origin, new_point - origin)
#         A, B, C = normal_vector
#         D = -np.dot(normal_vector, origin)
#         print(f"黄色平面的方程: {A:.4f}x + {B:.4f}y + {C:.4f}z + {D:.4f} = 0")
#
#         # 构造黄色平面点云（放大）
#         x_range = np.linspace(-3, 70, 100)  # 扩大范围至 -3 到 3
#         y_range = np.linspace(-50, 3, 100)  # 扩大范围至 -3 到 3
#         X, Y = np.meshgrid(x_range, y_range)
#         Z = (-A / C * X - B / C * Y - D / C)
#         Z_thickness = np.linspace(-0.05, 0.05, 5)  # 在厚度范围内生成多个层
#         plane_points_dense = np.vstack(
#             [np.column_stack((X.flatten(), Y.flatten(), Z.flatten() + t))
#              for t in Z_thickness]
#         )
#
#         plane_pcd_new = o3d.geometry.PointCloud()
#         plane_pcd_new.points = o3d.utility.Vector3dVector(plane_points_dense)
#         plane_pcd_new.paint_uniform_color([1, 1, 0])  # 新平面颜色：黄色
#         vis.add_geometry(plane_pcd_new)
#
#     # 启动可视化
#     vis.run()
#     vis.destroy_window()

import numpy as np
import open3d as o3d

# def visualize_with_open3d(point_cloud_np, plane_equations, plane_B_equation, thickness_A=0.01, thickness_B=0.1):
#     if point_cloud_np.size == 0:
#         print("点云数据为空，无法进行可视化。")
#         return
#
#     # 创建点云对象
#     pcd = o3d.geometry.PointCloud()
#     pcd.points = o3d.utility.Vector3dVector(point_cloud_np)
#     pcd.paint_uniform_color([0, 0, 1])  # 点云颜色：蓝色
#
#     # 创建一个可视化窗口
#     vis = o3d.visualization.Visualizer()
#     vis.create_window(window_name='Point Cloud and Planes', width=800, height=600)
#
#     # 设置视点参数
#     ctr = vis.get_view_control()
#     ctr.set_front([0, 0, -1])
#     ctr.set_up([0, 1, 0])
#     ctr.set_lookat([0, 0, 0])
#
#     # 添加点云
#     vis.add_geometry(pcd)
#
#     # 初始化最低点
#     _,lowest_point = move_plane_equation(plane_equation_A,  steps, point_cloud_np, plane_B_equation)
#
#     # 绘制平面A
#     for eq in plane_equations:
#         A, B, C, D = eq
#         # 调整网格范围以匹配点云范围
#         x_range = np.linspace(min(point_cloud_np[:, 0]), max(point_cloud_np[:, 0]), 10)
#         y_range = np.linspace(min(point_cloud_np[:, 1]), max(point_cloud_np[:, 1]), 10)
#         X, Y = np.meshgrid(x_range, y_range)
#         Z = (-A / C * X - B / C * Y - D / C)
#         plane_points = np.vstack((X.flatten(), Y.flatten(), Z.flatten())).T
#         plane_points[:, 2] += thickness_A
#         plane_pcd = o3d.geometry.PointCloud()
#         plane_pcd.points = o3d.utility.Vector3dVector(plane_points)
#         plane_pcd.paint_uniform_color([1, 0, 0])  # 平面A颜色：红色
#         vis.add_geometry(plane_pcd)
#
#     # 绘制平面B
#     A, B, C, D = plane_B_equation
#     x_range = np.linspace(min(point_cloud_np[:, 0]), max(point_cloud_np[:, 0]), 10)
#     y_range = np.linspace(min(point_cloud_np[:, 1]), max(point_cloud_np[:, 1]), 10)
#     X, Y = np.meshgrid(x_range, y_range)
#     Z = (-A / C * X - B / C * Y - D / C)
#     plane_points_B = np.vstack((X.flatten(), Y.flatten(), Z.flatten())).T
#     plane_points_B[:, 2] += thickness_B
#     plane_pcd_B = o3d.geometry.PointCloud()
#     plane_pcd_B.points = o3d.utility.Vector3dVector(plane_points_B)
#     plane_pcd_B.paint_uniform_color([0, 1, 0])  # 平面B颜色：绿色
#     vis.add_geometry(plane_pcd_B)
#
#     # 根据最低点生成新点，并可视化由 (0, 0, 0)、最低点和生成点构成的平面
#     if lowest_point is not None:
#         x_lowest, y_lowest, z_lowest = lowest_point
#         y_new = np.random.uniform(0, 1)
#         new_point = np.array([x_lowest, y_new, z_lowest])
#
#         # 打印三个点的坐标
#         print(f"构成黄色平面的三个点坐标：")
#         print(f"原点: (0, 0, 0)")
#         print(f"最低点: {lowest_point}")
#         print(f"新生成点: {new_point}")
#
#         # 构建新平面上的三个点
#         origin = np.array([0, 0, 0])
#         plane_points_new = np.array([origin, lowest_point, new_point])
#
#         # 计算新平面的法向量和平面方程
#         normal_vector = np.cross(lowest_point - origin, new_point - origin)
#         A, B, C = normal_vector
#         D = -np.dot(normal_vector, origin)
#         print(f"黄色平面的方程: {A:.4f}x + {B:.4f}y + {C:.4f}z + {D:.4f} = 0")
#
#         # 计算黄色平面与平面B的交点
#         # 设黄色平面方程为 A1*x + B1*y + C1*z + D1 = 0
#         # 平面B的方程为 A2*x + B2*y + C2*z + D2 = 0
#         A1, B1, C1, D1 = A, B, C, D
#         A2, B2, C2, D2 = plane_B_equation
#
#         # 通过解方程组求交线的Z坐标
#         # 两个平面方程：
#         # A1*x + B1*y + C1*z + D1 = 0
#         # A2*x + B2*y + C2*z + D2 = 0
#
#         # 我们可以用一个具体的x和y值，求解z的交点
#         x_intersect = 0  # 可以选择任意x值
#         y_intersect = 0  # 可以选择任意y值
#
#         # 解出z的值
#         z_intersect = (-A1 * x_intersect - B1 * y_intersect - D1) / C1
#         print(f"黄色平面与平面B交点的Z坐标值为：{z_intersect}")
#
#         # 构造黄色平面点云（放大）
#         x_range = np.linspace(-3, 70, 100)  # 扩大范围至 -3 到 3
#         y_range = np.linspace(-50, 3, 100)  # 扩大范围至 -3 到 3
#         X, Y = np.meshgrid(x_range, y_range)
#         Z = (-A / C * X - B / C * Y - D / C)
#         Z_thickness = np.linspace(-0.05, 0.05, 5)  # 在厚度范围内生成多个层
#         plane_points_dense = np.vstack(
#             [np.column_stack((X.flatten(), Y.flatten(), Z.flatten() + t))
#              for t in Z_thickness]
#         )
#
#         plane_pcd_new = o3d.geometry.PointCloud()
#         plane_pcd_new.points = o3d.utility.Vector3dVector(plane_points_dense)
#         plane_pcd_new.paint_uniform_color([1, 1, 0])  # 新平面颜色：黄色
#         vis.add_geometry(plane_pcd_new)
#a
#     # 启动可视化
#     vis.run()
#     vis.destroy_window()


# -------------------------------------------------------------------------------------------------

# #
# def visualize_with_open3d(point_cloud_np, plane_equations, plane_B_equation, thickness_A=0.01, thickness_B=0.1):
#     if point_cloud_np.size == 0:
#         print("点云数据为空，无法进行可视化。")
#         return
#
#     # 创建点云对象
#     pcd = o3d.geometry.PointCloud()
#     pcd.points = o3d.utility.Vector3dVector(point_cloud_np)
#     pcd.paint_uniform_color([0, 0, 1])  # 点云颜色：蓝色
#
#     # 创建一个可视化窗口
#     vis = o3d.visualization.Visualizer()
#     vis.create_window(window_name='Point Cloud and Planes', width=800, height=600)
#
#     # 添加点云
#     vis.add_geometry(pcd)
#
#     # 计算平面A的最低点
#     _, lowest_point_A = move_plane_equation(plane_equation_A, steps, point_cloud_np, plane_B_equation)
#
#     # 创建黄色平面，初始为水平
#     yellow_plane_points = create_plane_points(
#         [0, 0, 1, -lowest_point_A[2]],  # 假设平面方程 z = lowest_point_A[2]
#         point_cloud_np,
#         thickness=0,
#         angle_deg=0
#     )
#
#     yellow_plane_pcd = o3d.geometry.PointCloud()
#     yellow_plane_pcd.points = o3d.utility.Vector3dVector(yellow_plane_points)
#     yellow_plane_pcd.paint_uniform_color([1, 1, 0])  # 黄色
#     vis.add_geometry(yellow_plane_pcd)
#
#     # 移动黄色平面到最低点并进行倾斜
#     origin_to_lowest_vector = lowest_point_A  # 原点到最低点的向量
#     yellow_plane_normal = np.array([0, 0, 1])  # 水平面法向量
#
#     # 计算旋转后的法向量，使得黄色平面法向量与最低点的向量垂直
#     new_yellow_normal = rotate_normal_vector_to_perpendicular(yellow_plane_normal, origin_to_lowest_vector)
#
#     # 计算平面常数D，确保平面通过最低点
#     D = -np.dot(new_yellow_normal, lowest_point_A)  # 平面方程常数项
#
#     # 打印倾斜后的黄色平面方程
#     print(f"倾斜后的黄色平面方程: {new_yellow_normal[0]:.4f}x + {new_yellow_normal[1]:.4f}y + {new_yellow_normal[2]:.4f}z + {D:.4f}")
#
#     # 更新黄色平面的点，使用新的法向量
#     tilted_yellow_plane_points = create_plane_points(
#         np.hstack((-new_yellow_normal, D)),  # 使用新的法向量和D值
#         point_cloud_np,
#         thickness=1,
#         angle_deg=0
#     )
#
#     tilted_yellow_plane_pcd = o3d.geometry.PointCloud()
#     tilted_yellow_plane_pcd.points = o3d.utility.Vector3dVector(tilted_yellow_plane_points)
#     tilted_yellow_plane_pcd.paint_uniform_color([1, 1, 0])  # 黄色
#     # vis.add_geometry(tilted_yellow_plane_pcd)
#
#     # 绘制平面A和B
#     for eq in plane_equations:
#         plane_points_A = create_plane_points(eq, point_cloud_np, thickness_A, angle_deg=0)
#         plane_pcd_A = o3d.geometry.PointCloud()
#         plane_pcd_A.points = o3d.utility.Vector3dVector(plane_points_A)
#         plane_pcd_A.paint_uniform_color([1, 0, 0])  # 平面A颜色：红色
#         # vis.add_geometry(plane_pcd_A)
#
#     A, B, C, D = plane_B_equation
#     plane_points_B = create_plane_points(plane_B_equation, point_cloud_np, thickness_B, angle_deg=0)
#     plane_pcd_B = o3d.geometry.PointCloud()
#     plane_pcd_B.points = o3d.utility.Vector3dVector(plane_points_B)
#     plane_pcd_B.paint_uniform_color([0, 1, 0])  # 平面B颜色：绿色
#     # vis.add_geometry(plane_pcd_B)
#
#     # 启动可视化
#     vis.run()
#     vis.update_geometry(tilted_yellow_plane_pcd)
#     vis.destroy_window()
import random

def visualize_with_open3d(point_cloud_np, plane_equations, plane_B_equation, thickness_A=0.01, thickness_B=0.1,
                          point_size=0.05):
    if point_cloud_np.size == 0:
        print("点云数据为空，无法进行可视化。")
        return

    # 创建点云对象
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_cloud_np)
    pcd.paint_uniform_color([0, 0, 1])  # 点云颜色：蓝色

    # 创建一个可视化窗口
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name='Point Cloud and Planes', width=800, height=600)

    # 添加点云
    # vis.add_geometry(pcd)

    # 计算平面A的最低点
    _, lowest_point_A = move_plane_equation(plane_equation_A, steps, point_cloud_np, plane_B_equation)

    # 增加最低点尺寸（生成一个球形点云）
    num_points = 100  # 控制生成的点的数量
    sphere_radius = point_size  # 设置球的半径

    # 生成一个球形区域中的点
    sphere_points = []
    for _ in range(num_points):
        # 随机生成球面上的点
        phi = np.random.uniform(0, 2 * np.pi)
        theta = np.random.uniform(0, np.pi)
        x = lowest_point_A[0] + sphere_radius * np.sin(theta) * np.cos(phi)
        y = lowest_point_A[1] + sphere_radius * np.sin(theta) * np.sin(phi)
        z = lowest_point_A[2] + sphere_radius * np.cos(theta)
        sphere_points.append([x, y, z])

    sphere_points = np.array(sphere_points)

    # 创建球形点云
    sphere_pcd = o3d.geometry.PointCloud()
    sphere_pcd.points = o3d.utility.Vector3dVector(sphere_points)
    sphere_pcd.paint_uniform_color([1, 0, 0])  # 红色表示最低点区域
    # vis.add_geometry(sphere_pcd)


    # 创建平面E并初始化为水平
    plane_E_points = create_plane_points(
        [0, 0, 1, -lowest_point_A[2]],  # 假设平面方程 z = lowest_point_A[2]
        point_cloud_np,
        thickness=0,
        angle_deg=0
    )
    plane_E_pcd = o3d.geometry.PointCloud()
    plane_E_pcd.points = o3d.utility.Vector3dVector(plane_E_points)
    plane_E_pcd.paint_uniform_color([1, 1, 0])  # 黄色表示平面E
    # vis.add_geometry(plane_E_pcd)

    A, B, C, D = [0, 0, 1, -lowest_point_A[2]]  # 平面E的方程是 z = lowest_point_A[2]
    plane_E_equation = [A, B, C, D]
    print(f"平面E方程: {A:.4f}x + {B:.4f}y + {C:.4f}z + {D:.4f} = 0")

    # 计算平面B的距离函数
    def point_to_plane_distance(point, plane_eq):
        A, B, C, D = plane_eq
        numerator = abs(A * point[0] + B * point[1] + C * point[2] + D)
        denominator = np.sqrt(A ** 2 + B ** 2 + C ** 2)
        return numerator / denominator

    # 计算平面A最低点到平面B的距离
    distance_lowest_to_B = point_to_plane_distance(lowest_point_A, plane_B_equation)
    print(f"平面A最低点到平面B的距离: {distance_lowest_to_B}")

    # --------------------------------------------------------------------------------------------------

    def line_intersection_of_planes(plane_E, plane_F):
        A1, B1, C1, D1 = plane_E
        A2, B2, C2, D2 = plane_F

        # 求解交线的参数化方程
        # 解方程组
        # 使用 np.linalg.solve 解这个方程组

        # 设定一个自由变量 (比如设定 x = 0，解 y 和 z)
        A = np.array([[A1, B1, C1], [A2, B2, C2]])
        B = np.array([-D1, -D2])

        # 检查是否可以解这个方程组
        if np.linalg.matrix_rank(A) < 2:
            print("平面不相交或重合")
            return None, None

        # 解方程组
        solution = np.linalg.lstsq(A, B, rcond=None)[0]
        point_on_line = solution

        # 计算交线的方向
        direction = np.cross([A1, B1, C1], [A2, B2, C2])

        return point_on_line, direction

    # 计算平面F的方程
    normal_B = np.array(plane_B_equation[:3])  # 平面B的法向量
    F_constant = -(np.dot(normal_B, lowest_point_A))  # 平面F的常数项
    plane_F_equation = np.hstack((normal_B, F_constant))

    # 打印平面F的方程
    print(
        f"平面F方程: {plane_F_equation[0]:.4f}x + {plane_F_equation[1]:.4f}y + {plane_F_equation[2]:.4f}z + {plane_F_equation[3]:.4f} = 0")

    # 获取平面E和平面F的交线
    point_on_line, direction = line_intersection_of_planes([0, 0, 1, -lowest_point_A[2]], plane_F_equation)

    print(f"交线的方向向量: {direction}")
    # 确保 point_s 已初始化
    point_s = point_on_line

    # 使用交点和方向来确定交线的两个端点
    # 选择一段适当的长度，作为交线的显示范围
    line_length = 30  # 可调整显示的交线长度

    point_s1 = point_on_line - line_length * direction
    point_s2 = point_on_line + line_length * direction

    # 可视化交线
    intersection_points = [point_s1, point_s2]
    lines = [[0, 1]]  # 连接点0和点1
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(intersection_points)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.paint_uniform_color([1, 0, 0])  # 红色表示交线
    # vis.add_geometry(line_set)

    # 确保 point_s 不和平面A的最低点重合
    while np.allclose(point_s, lowest_point_A):  # 检查是否与最低点重合
        random_param = random.uniform(0, 1)
        point_s = point_on_line + random_param * direction

    print(f"平面E和平面F交线上的随机点: {point_s}")

    # ------------------------------------------------------------------------------------------------

    # 使用原点、最低点和平面E点生成平面G
    origin = np.array([0, 0, 0])
    v1 = lowest_point_A - origin
    v2 = point_s - origin
    normal_G = np.cross(v1, v2)  # 计算法向量

    # 确保法向量的方向正确
    if np.dot(normal_G, lowest_point_A) < 0:  # 如果法向量与最低点A的方向不一致，则反转法向量
        normal_G = -normal_G

    normal_G = normal_G / np.linalg.norm(normal_G)  # 单位化
    G_constant = -np.dot(normal_G, origin)  # 计算常数G
    plane_G_equation = np.hstack((normal_G, G_constant))

    # 打印平面G方程
    print(
        f"平面G方程: {plane_G_equation[0]:.4f}x + {plane_G_equation[1]:.4f}y + {plane_G_equation[2]:.4f}z + {plane_G_equation[3]:.4f}")
    print(f"平面G法向量: {normal_G}")

    # 确保平面G包含最低点A
    assert np.isclose(np.dot(normal_G, lowest_point_A) + G_constant, 0), "平面G没有通过最低点A！"

    # 创建平面G点
    plane_G_points = create_plane_points(plane_G_equation, point_cloud_np, thickness=0.01, angle_deg=0)
    plane_G_pcd = o3d.geometry.PointCloud()
    plane_G_pcd.points = o3d.utility.Vector3dVector(plane_G_points)
    plane_G_pcd.paint_uniform_color([0, 0, 0])  # 黑色表示平面G
    # vis.add_geometry(plane_G_pcd)

    # 修改平面F的Z方向显示长度为7个单位
    thickness_F = 7.0  # 设置平面F的Z方向长度为7个单位
    plane_F_points = create_plane_points(plane_F_equation, point_cloud_np, thickness=thickness_F, angle_deg=0)
    plane_F_pcd = o3d.geometry.PointCloud()
    plane_F_pcd.points = o3d.utility.Vector3dVector(plane_F_points)
    plane_F_pcd.paint_uniform_color([0, 0, 0])  # 黑色表示平面F
    # vis.add_geometry(plane_F_pcd)

    # 可视化交线上的随机点 (黑色点)
    point_s_pcd = o3d.geometry.PointCloud()
    point_s_pcd.points = o3d.utility.Vector3dVector([point_s])  # 将随机点添加为点云
    point_s_pcd.paint_uniform_color([0, 0, 0])  # 黑色表示该点
    # vis.add_geometry(point_s_pcd)

    # 可视化平面E和平面F的交线 (红色线)
    intersection_points = [point_on_line, point_s]
    lines = [[0, 1]]  # 连接点0和点1
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(intersection_points)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.paint_uniform_color([1, 0, 0])  # 红色表示交线
    # vis.add_geometry(line_set)

    # 可视化所有平面
    for eq in plane_equations:
        plane_points_A = create_plane_points(eq, point_cloud_np, thickness_A, angle_deg=0)
        plane_pcd_A = o3d.geometry.PointCloud()
        plane_pcd_A.points = o3d.utility.Vector3dVector(plane_points_A)
        plane_pcd_A.paint_uniform_color([1, 0, 0])  # 平面A颜色：红色

    plane_points_B = create_plane_points(plane_B_equation, point_cloud_np, thickness_B, angle_deg=0)
    plane_pcd_B = o3d.geometry.PointCloud()
    plane_pcd_B.points = o3d.utility.Vector3dVector(plane_points_B)
    plane_pcd_B.paint_uniform_color([0, 1, 0])  # 平面B颜色：绿色

    # 打印平面F的方程
    print(
        f"平面B方程: {plane_B_equation[0]:.4f}x + {plane_B_equation[1]:.4f}y + {plane_B_equation[2]:.4f}z + {plane_B_equation[3]:.4f} = 0")

    # 启动可视化
    # vis.run()
    # vis.destroy_window()

    return plane_G_equation

def visualize_and_filter_point_cloud():
    """
    可视化点云，并对点云进行XYZ方向上的滤波和基于平面方程的滤波。
    滤波范围和方程参数在函数内部定义，点云路径也硬编码在函数体内。
    """
    # 点云文件路径
    # point_cloud_path = pcd_chuanwei  # 替换为你实际的点云文件路径

    # 加载点云文件
    # pcd = o3d.io.read_point_cloud(point_cloud_path)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pcd_chuanwei)

    # 检查点云是否为空
    if len(pcd.points) == 0:
        print("Original point cloud is empty!")
        return

    # 可视化原始点云
    # o3d.visualization.draw_geometries([pcd], window_name="Original Point Cloud")

    # 点云转换为numpy数组
    points = np.asarray(pcd.points)

    # XYZ方向上的滤波范围
    x_range = (46, 100)  # 设置X轴的滤波范围
    y_range = (-40, 2)  # 设置Y轴的滤波范围
    z_range = (-500, 500)  # 设置Z轴的滤波范围

    # XYZ方向的滤波
    points = points[(points[:, 0] >= x_range[0]) & (points[:, 0] <= x_range[1])]
    points = points[(points[:, 1] >= y_range[0]) & (points[:, 1] <= y_range[1])]
    points = points[(points[:, 2] >= z_range[0]) & (points[:, 2] <= z_range[1])]

    print(f"Number of points after XYZ filtering: {points.shape[0]}")

    # 平面方程参数 (A, B, C, D) 示例：平面 z = 0  =>  0*x + 0*y + 1*z + 0 = 0
    plane_eq = visualize_with_open3d(point_cloud_np, moved_plane_equations, plane_B_equation, thickness_A, thickness_B)  # 平面方程：z = 0
    A, B, C, D = plane_eq

    # 计算每个点到平面的距离，保留距离平面方程满足条件的点
    distances = A * points[:, 0] + B * points[:, 1] + C * points[:, 2] + D
    points = points[distances <0]

    print(f"Number of points after plane filtering: {points.shape[0]}")

    # 如果点云为空，打印并退出
    if points.shape[0] == 0:
        print("Filtered point cloud is empty!")
        return

    # 创建新的点云对象
    filtered_pcd = o3d.geometry.PointCloud()
    filtered_pcd.points = o3d.utility.Vector3dVector(points)

    return np.asarray(filtered_pcd.points)

    # 可视化滤波后的点云
    # o3d.visualization.draw_geometries(
    #     [filtered_pcd],
    #     window_name="Filtered Point Cloud",
    #     width=800,
    #     height=600
    # )



def create_plane_points(plane_equation, point_cloud_np, thickness, angle_deg):
    """
    根据平面方程、点云范围和角度创建平面网格点
    """
    A, B, C, D = plane_equation

    # 调整网格范围以匹配点云范围
    x_range = np.linspace(min(point_cloud_np[:, 0]), max(point_cloud_np[:, 0]), 10)
    y_range = np.linspace(min(point_cloud_np[:, 1]), max(point_cloud_np[:, 1]), 10)
    X, Y = np.meshgrid(x_range, y_range)

    # 检查 C 是否为零
    if C == 0:
        print("警告：C值为零，平面可能是水平的。")
        # 如果C为零，直接设定Z值为常数（例如，D/A），或者根据具体情况调整
        Z = np.full(X.shape, -D / A if A != 0 else 0)  # 如果A也是零，Z值设置为0，或根据具体情况调整
    else:
        # 计算Z值
        Z = (-A / C * X - B / C * Y - D / C)

    # 如果希望通过角度控制平面的倾斜，调整法向量
    # 将法向量的方向旋转相应的角度
    angle_rad = np.radians(angle_deg)
    rotation_matrix = np.array([
        [1, 0, 0],
        [0, np.cos(angle_rad), -np.sin(angle_rad)],
        [0, np.sin(angle_rad), np.cos(angle_rad)]
    ])

    normal_vector = np.array([A, B, C])
    rotated_normal = rotation_matrix @ normal_vector

    # 检查旋转后的法向量 z 分量是否为零
    if rotated_normal[2] == 0:
        print("警告：旋转后的法向量Z分量为零，平面可能是水平的。")
        Z_rotated = np.full(X.shape, -D / rotated_normal[0] if rotated_normal[0] != 0 else 0)
    else:
        # 使用旋转后的法向量来调整Z值的计算
        Z_rotated = (-rotated_normal[0] / rotated_normal[2] * X - rotated_normal[1] / rotated_normal[2] * Y - D / rotated_normal[2])

    # 创建平面点
    plane_points = np.vstack((X.flatten(), Y.flatten(), Z_rotated.flatten())).T

    return plane_points

# -----------------------------------------------------------------------------------------------




# 现在在调用 move_plane_equation 函数时，传递 point_cloud_np
# moved_plane_equations,_ = move_plane_equation(plane_equation_A,  steps, point_cloud_np, plane_B_equation)

# 在 visualize_with_open3d 函数中，确保绘制平面时使用 numpy 数组
# visualize_with_open3d(point_cloud_np, moved_plane_equations, plane_B_equation, thickness_A, thickness_B)

# visualize_and_filter_point_cloud()

def cut_rail(a,b,c,d, point_tail_np,point_cloud_np1):

    global point_cloud_np
    global plane_B_equation
    global pcd_chuanwei
    print(f'平面是:{[a,b,c,d]}')
    print(point_tail_np)
    print(point_cloud_np1)
    point_cloud_np = point_cloud_np1
    plane_B_equation = [a,b,c,d]
    pcd_chuanwei = point_tail_np

    # 现在在调用 move_plane_equation 函数时，传递 point_cloud_np
    moved_plane_equations, _ = move_plane_equation(plane_equation_A, steps, point_cloud_np, plane_B_equation)

    # 在 visualize_with_open3d 函数中，确保绘制平面时使用 numpy 数组
    visualize_with_open3d(point_cloud_np, moved_plane_equations, plane_B_equation, thickness_A, thickness_B)

    return visualize_and_filter_point_cloud()