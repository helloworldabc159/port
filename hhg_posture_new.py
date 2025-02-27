import matplotlib.pyplot as plt
from ultralytics import YOLO
from PIL import Image
import cv2
import math
import scipy.io
import numpy as np
import open3d as o3d


##########################################################################################
def read_pcd_with_intensity(file_path):
    # 读取PCD文件并保留无效点
    pcd = o3d.io.read_point_cloud(file_path, remove_nan_points=True, remove_infinite_points=True)
    # print(np.asarray(pcd.points).shape)

    # 打开PCD文件并读取header
    with open(file_path, 'rb') as file:
        header = []
        while True:
            line = file.readline().strip()
            header.append(line)
            if line.startswith(b'DATA'):
                data_start = file.tell()
                break

    # 解析header以获取字段信息
    fields = None
    size = None
    type = None
    count = None
    data_format = None
    for line in header:
        if line.startswith(b'FIELDS'):
            fields = line.split()[1:]
        if line.startswith(b'SIZE'):
            size = list(map(int, line.split()[1:]))
        if line.startswith(b'TYPE'):
            type = line.split()[1:]
        if line.startswith(b'COUNT'):
            count = list(map(int, line.split()[1:]))
        if line.startswith(b'DATA'):
            data_format = line.split()[1].decode('utf-8')

    if fields is None:
        raise ValueError("No FIELDS information found in the PCD file.")
    if size is None or type is None or count is None:
        raise ValueError("Missing SIZE, TYPE, or COUNT information in the PCD file.")

    # 确定intensity在字段中的位置
    if b'intensity' not in fields:
        raise ValueError("No intensity information found in the PCD file.")
    intensity_index = fields.index(b'intensity')

    # 处理二进制格式的数据
    if data_format == 'binary':
        dtype = []
        for field, s, t, c in zip(fields, size, type, count):
            if t == b'F':
                dtype.append((field.decode('utf-8'), np.float32))
            elif t == b'U':
                dtype.append((field.decode('utf-8'), np.uint32))
            elif t == b'I':
                dtype.append((field.decode('utf-8'), np.int32))
            else:
                raise ValueError(f"Unsupported data type: {t}")

        # 重新打开文件并从数据部分读取
        with open(file_path, 'rb') as file:
            file.seek(data_start)
            data = np.fromfile(file, dtype=dtype)

        # 移除包含NaN和无穷大的点
        valid_mask = np.isfinite(data['x']) & np.isfinite(data['y']) & np.isfinite(data['z'])
        valid_mask &= (data['x'] != 0) | (data['y'] != 0) | (data['z'] != 0)
        for field in fields:
            valid_mask &= np.isfinite(data[field.decode('utf-8')])

        data = data[valid_mask]
        intensities = data['intensity']
    else:
        raise ValueError(f"Unsupported DATA format: {data_format}")

    # 创建一个新的点云对象并填充有效的点
    valid_points = np.vstack((data['x'], data['y'], data['z'])).T
    pcd.points = o3d.utility.Vector3dVector(valid_points)
    intensities = intensities.reshape(intensities.shape[0], 1)
    return pcd, intensities


def lslidar_to_IntensityView(points, res=0.1, img_x_range=(-10., 10.), img_y_range=(-10., 10.)):
    """
    功能：创建一张点云数据的强度图
    参数:
        points:     (numpy array)
                    点云数据的N列; 三列数据对应点云的x,y,z,intensity
        res:        (float)
                    所需的分辨率。每个输出像素将表示大小为正方形区域 res*res。
        img_x_range: (tuple of two floats)
                    (left, right)强度图的矩形的左右限制
        img_y_range:  (tuple of two floats)
                    (low, high)强度图的矩形高低限制
    返回值:
       表示强度图图像的 2D numpy 数组。
    """
    # 提取每个轴的点
    x_points = points[:, 0]
    y_points = points[:, 1]
    z_points = points[:, 2]
    i_points = points[:, 3]

    # FILTER - 只返回所需立方体内点的索引
    # 三个过滤器：从前到后、从左到右和高度范围
    # 注意左侧是激光雷达坐标中的正y轴
    x_filter = np.logical_and((x_points > img_x_range[0]), (x_points < img_x_range[1]))
    z_filter = np.logical_and((z_points > img_y_range[0]), (z_points < img_y_range[1]))
    filter = np.logical_and(x_filter, z_filter)
    indices = np.argwhere(filter).flatten()

    # KEEPERS
    x_points = x_points[indices]
    y_points = y_points[indices]
    z_points = z_points[indices]
    i_points = i_points[indices]

    # 转换为像素位置值 - 基于分辨率
    x_img = (x_points / res).astype(np.int32)
    y_img = (-z_points / res).astype(np.int32)

    # 将像素移位为最小值为 （0，0）设立坐标原点
    # floor 和 ceil 用于防止移动后任何内容四舍五入到 0 以下
    x_img -= int(np.floor(img_x_range[0] / res))
    y_img += int(np.ceil(img_y_range[1] / res))

    cmap = plt.colormaps.get_cmap('hsv')
    colors = cmap(i_points / np.max(i_points))
    print(colors.shape)
    rgb = tuple(map(tuple, np.array(colors[:, :3] * 255).astype(int)))

    # 初始化空数组，以达到想要的维度，即生成图片的大小
    x_max = 1 + int((img_x_range[1] - img_x_range[0]) / res)
    y_max = 1 + int((img_y_range[1] - img_y_range[0]) / res)
    im = np.zeros([y_max, x_max, 3], dtype=np.uint8)

    # 将像素值填充到数组里
    im[y_img, x_img] = rgb

    return im


def restore_on_pointCloud(original_pointCloud, real_coordinates,
                          res=0.01, img_x_range=(-10., 10.), img_y_range=(-10., 10.)):
    """
    功能：将锚框复原在点云上，并用open3d进行展示
    参数:
        original_pointCloud:(numpy array)
                            原始点云数据的N列; 三列数据对应点云的x,y,z
        real_coordinate:    (tuple of four floats)
                            真实角点坐标，x1,x2,y1,y2
        res:                (float)
                            所需的分辨率。每个输出像素将表示大小为正方形区域 res*res。
        img_x_range: (tuple of two floats)
                    (left, right)强度图的矩形的左右限制
        img_y_range:  (tuple of two floats)
                    (low, high)强度图的矩形高低限制
    返回值:
        角点的3D坐标
    """
    # 准备原始点云
    pcd_1 = o3d.geometry.PointCloud()
    pcd_1.points = o3d.utility.Vector3dVector(original_pointCloud)
    pcd_1.paint_uniform_color([0, 0.706, 1])  # 上色区分
    edge_pointclouds = []
    row_points = []
    for real_coordinate in real_coordinates:
        x = img_x_range[0] + real_coordinate[1] * res
        y = img_y_range[1] - real_coordinate[0] * res
        edge_pointcloud = [x, 0., y]
        row_point = [x, 0., y]
        edge_pointclouds.append(edge_pointcloud)
        row_points.append(row_point)

    # 用于计算俯仰角的向量
    row_points = np.array(row_points)
    row_vector = row_points[1] - row_points[0]
    print('row_vector', row_vector)
    # 边界角点绘制

    print(f"角点\n{edge_pointclouds}")
    pcd_2 = o3d.geometry.PointCloud()
    pcd_2.points = o3d.utility.Vector3dVector(edge_pointclouds)
    pcd_2.paint_uniform_color([1, 0.293, 0])
    o3d.visualization.draw_geometries([pcd_1] + [pcd_2] + [axis_pcd], window_name='Open3D', width=1080,
                                      height=800, left=50, top=50, point_show_normal=False, mesh_show_wireframe=False,
                                      mesh_show_back_face=False)
    return edge_pointclouds, row_vector


def edge_estimate(points, max_nn=30, normals_raduis=10, search_raduis=10, nb_points=30, angle=90, vis=False):
    """
    :param points: 传入的点云点坐标，np格式
    :param max_nn: 邻域内最大值
    :param normals_raduis:用于HybridSearch的邻域内搜索半径
    :param search_raduis:边界提取搜索半径
    :param nb_points:边界提取领域点
    :param angle:角度阈值
    :param vis:是否可视化
    :return:
    """

    pcd = o3d.t.geometry.PointCloud(points)
    pcd.estimate_normals(max_nn=max_nn, radius=normals_raduis)  # 计算点云法向量
    boundarys, mask = pcd.compute_boundary_points(radius=search_raduis, max_nn=nb_points,
                                                  angle_threshold=angle)  # 边界提取的搜索半径、邻域最大点数和夹角阈值（角度制）

    print(f"Detect {boundarys.point.positions.shape[0]} boundary points from {pcd.point.positions.shape[0]} points.")
    boundarys = boundarys.paint_uniform_color([1.0, 0.0, 0.0])
    boundarys_points = np.array(boundarys.to_legacy().points)

    if vis == True:
        pcd.paint_uniform_color([0.6, 0.6, 0.6])

        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name='三维点云边界提取', width=1200, height=800)
        # 可视化参数设置
        opt = vis.get_render_option()
        opt.background_color = np.asarray([1, 1, 1])  # 设置背景色
        opt.point_size = 3  # 设置点的大小
        vis.add_geometry(boundarys.to_legacy())  # 加载边界点云到可视化窗口
        vis.add_geometry(pcd.to_legacy())  # 加载原始点云到可视化窗口
        vis.run()  # 激活显示窗口，这个函数将阻塞当前线程，直到窗口关闭。
        vis.destroy_window()  # 销毁窗口，这个函数必须从主线程调用。

    return boundarys_points


def get_line_equation(x1, y1, x2, y2):
    A = y2 - y1
    B = x1 - x2
    C = x2 * y1 - x1 * y2
    return A, B, C


def main(points):
    if len(points) != 4:
        raise ValueError("必须提供四个点的坐标")

    line_equations = []

    for i in range(4):
        x1, y1 = points[i]
        x2, y2 = points[(i + 1) % 4]
        A, B, C = get_line_equation(x1, y1, x2, y2)
        line_equations.append((A, B, C))

    return line_equations


def find_corners(edge_points):
    """
    计算并返回给定边界点列表的四个顶点。

    参数:
    edge_points (list of tuples): 图像中目标的边界点列表，每个点是 (x, y) 的元组。

    返回:
    dict: 一个包含四个顶点的字典，键分别是 'top_left', 'top_right', 'bottom_left', 'bottom_right'。
    """

    # 计算中心点
    center_x = sum(x for x, y in edge_points) / len(edge_points)
    center_y = sum(y for x, y in edge_points) / len(edge_points)

    # 将点按相对位置分组
    top_left_group = [(x, y) for x, y in edge_points if x < center_x and y < center_y]
    top_right_group = [(x, y) for x, y in edge_points if x >= center_x and y < center_y]
    bottom_left_group = [(x, y) for x, y in edge_points if x < center_x and y >= center_y]
    bottom_right_group = [(x, y) for x, y in edge_points if x >= center_x and y >= center_y]

    # 在每组中选择离中心点最远的点作为顶点
    top_left = max(top_left_group, key=lambda p: math.hypot(p[0] - center_x, p[1] - center_y))
    top_right = max(top_right_group, key=lambda p: math.hypot(p[0] - center_x, p[1] - center_y))
    bottom_left = max(bottom_left_group, key=lambda p: math.hypot(p[0] - center_x, p[1] - center_y))
    bottom_right = max(bottom_right_group, key=lambda p: math.hypot(p[0] - center_x, p[1] - center_y))

    return {
        'top_left': top_left,
        'top_right': top_right,
        'bottom_left': bottom_left,
        'bottom_right': bottom_right
    }


def MSAC(data):
    number = data.shape[1]  # 点的总数
    iter = 10000  # 迭代次数
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


def ensure_direction(vector, reference=np.array([0, 1, 0])):
    # 规范化向量
    vector_normalized = vector / np.linalg.norm(vector)
    reference_normalized = reference / np.linalg.norm(reference)
    # 如果点积为负，反转 vector 的方向
    if np.dot(vector_normalized, reference_normalized) < 0:
        vector_normalized = -vector_normalized
    return vector_normalized


def yaw_pitch_row(normal_vector, projection_plane='yaw'):
    if projection_plane == 'yaw':
        # xoy 平面的法向量
        plane_normal = np.array([0, 0, 1])
        axis_vector = np.array([0, 1, 0])  # y 轴
    elif projection_plane == 'pitch':
        # yoz 平面的法向量
        plane_normal = np.array([1, 0, 0])
        axis_vector = np.array([0, 1, 0])  # y 轴
    elif projection_plane == 'roll':
        # xoz 平面的法向量
        plane_normal = np.array([0, 1, 0])
        axis_vector = np.array([1, 0, 0])  # x 轴
    else:
        raise ValueError("Invalid projection plane. Choose 'xoz' or 'yoz'.")

    # 计算法向量在投影平面上的投影向量
    projection_vector = normal_vector - np.dot(normal_vector, plane_normal) / np.linalg.norm(
        plane_normal) ** 2 * plane_normal

    # 计算投影向量与轴的夹角
    cos_angle = np.dot(projection_vector, axis_vector) / (
                np.linalg.norm(projection_vector) * np.linalg.norm(axis_vector))
    angle = np.arccos(cos_angle)

    # 确定角度的符号
    if projection_plane == 'yaw':
        # 如果投影向量在 y 轴左侧（即 x 轴正方向），角度为正；否则为负
        if projection_vector[0] >= 0:
            return np.degrees(angle)
        else:
            return -np.degrees(angle)
    elif projection_plane == 'pitch':
        # 如果投影向量在 y 轴以上（即 z 轴正方向），角度为正；否则为负
        if projection_vector[2] >= 0:
            return np.degrees(angle)
        else:
            return -np.degrees(angle)
    elif projection_plane == 'roll':
        # 如果投影向量在 x 轴以上（即 z 轴正方向），角度为正；否则为负
        if projection_vector[2] >= 0:
            return np.degrees(angle)
        else:
            return -np.degrees(angle)


if __name__ == "__main__":
    # 路径读取
    model = YOLO(r'HHG_intensity.pt')  # load a custom model
    # img_path = r"E:\PythonProject\ultralytics-main(yolov8)\Intensity_Image\20240606160120.png"  # 替换为你的图片路径
    cloudpoint_path = r"H:\lizehui\originData\20240714164937.pcd"
    cloudpoint_path = r"HHG1111.pcd"
    # 超参数设置
    res = 0.02
    img_x_range = (-6., 5.)
    img_y_range = (-3., 3.)
    # 数据读取
    # image = cv2.imread(img_path)
    pcd, intensities = read_pcd_with_intensity(cloudpoint_path)
    points = np.hstack((np.asarray(pcd.points), intensities))
    image = lslidar_to_IntensityView(points=points, res=res, img_x_range=img_x_range, img_y_range=img_y_range)
    # 识别来自文件夹的图像
    # results = model.predict(source="test/pics", ……)
    # 识别来自摄像头的图像
    # results = model.predict(source="0", ……)
    # 识别结果
    results = model.predict(source=image, save=False, save_txt=False, show_boxes=False)
    # 识别结果可视化
    annotated_frame = results[0].plot()
    cv2.imshow("YOLOv8 Inference", annotated_frame)
    cv2.waitKey(0)
    # -----------------------------------点云处理--------------------------------
    axis_pcd = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1, origin=[0, 0, 0])

    # 供静态测试使用
    # 点云文件要求（x,y,z,intensity）二进制PCD文件
    points = np.hstack((np.asarray(pcd.points), intensities))
    # 实时测试代码（待补充）

    # 点云生成深度图
    # image = lslidar_to_IntensityView(points=points, res=0.01, img_x_range=(-6., 5.), img_y_range=(-3., 3.))
    # 计算角点坐标
    pixel_xy = results[0].masks.xy[0]  # masks.xy[n]: n为图中第n个检测目标
    edge_points = np.array(pixel_xy, np.int32)
    corners = find_corners(edge_points)
    top_left_x, top_left_y = corners['top_left']
    top_right_x, top_right_y = corners['top_right']
    bottom_left_x, bottom_left_y = corners['bottom_left']
    bottom_right_x, bottom_right_y = corners['bottom_right']

    real_coordinates = [[top_left_y, top_left_x, image[top_left_y, top_left_x]],
                        [top_right_y, top_right_x, image[top_right_y, top_right_x]],
                        [bottom_right_y, bottom_right_x, image[bottom_right_y, bottom_right_x]],
                        [bottom_left_y, bottom_left_x, image[bottom_left_y, bottom_left_x]]]

    print('2D角点', real_coordinates)

    # 角点映射（2D转3D） 仅在xoz平面上，不计较深度值
    edge_points, row_vector = restore_on_pointCloud(np.asarray(pcd.points), real_coordinates=real_coordinates,
                                                    res=res, img_x_range=img_x_range, img_y_range=img_y_range)

    # 计算角点在雷达坐标系下的直线方程系数
    x_min = get_line_equation(edge_points[0][0], edge_points[0][2], edge_points[3][0], edge_points[3][2])
    print(x_min)  # 1&4
    x_max = get_line_equation(edge_points[1][0], edge_points[1][2], edge_points[2][0], edge_points[2][2])
    print(x_max)  # 2&3
    z_min = get_line_equation(edge_points[3][0], edge_points[3][2], edge_points[2][0], edge_points[2][2])
    print(z_min)  # 4&3
    z_max = get_line_equation(edge_points[0][0], edge_points[0][2], edge_points[1][0], edge_points[1][2])
    print(z_max)  # 1&2

    # 过滤出ROI内的点云
    points = np.asarray(pcd.points)
    new_pcd = o3d.geometry.PointCloud()
    after_filter = np.logical_and.reduce(
        [
            x_min[0] * points[:, 0] + x_min[1] * points[:, 2] + x_min[2] <= 0,
            x_max[0] * points[:, 0] + x_max[1] * points[:, 2] + x_max[2] >= 0,
            z_min[0] * points[:, 0] + z_min[1] * points[:, 2] + z_min[2] <= 0,
            z_max[0] * points[:, 0] + z_max[1] * points[:, 2] + z_max[2] >= 0
        ]
    )

    # 分割点云展示
    new_pcd.points = o3d.utility.Vector3dVector(
        points[after_filter]
    )
    print(np.asarray(new_pcd.points).shape)
    new_pcd.paint_uniform_color([0, 0, 0])
    o3d.visualization.draw_geometries([new_pcd, pcd, axis_pcd], window_name='Open3D', width=1080,
                                      height=800, left=50, top=50, point_show_normal=False, mesh_show_wireframe=False,
                                      mesh_show_back_face=False)
    # -----------------------------------拟合平面---------------------------------------
    # 获取新点云
    points = np.asarray(new_pcd.points).T  # 转换为MSAC所需的格式
    # 使用MSAC算法拟合平面并去除外点
    params, inliers = MSAC(points)
    # 计算拟合平面质点坐标（即位置坐标）
    centroid = np.mean(inliers, axis=1)
    print('质点坐标：', centroid)
    # 将内点转换回点云
    inlier_pcd = o3d.geometry.PointCloud()
    inlier_pcd.points = o3d.utility.Vector3dVector(inliers.T)
    # 绘制结果
    # 创建质心的小球
    centroid_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.01)
    centroid_sphere.translate(centroid)
    centroid_sphere.paint_uniform_color([1, 0, 0])  # 质心上色为红色
    inlier_pcd.paint_uniform_color([0, 1, 0])  # 内点上色为绿色
    o3d.io.write_point_cloud("inlier_pcd.pcd", inlier_pcd)
    o3d.visualization.draw_geometries([inlier_pcd, centroid_sphere, axis_pcd], window_name='Open3D', width=1080,
                                      height=800, left=50, top=50, point_show_normal=False, mesh_show_wireframe=False,
                                      mesh_show_back_face=False)
    # 打印平面方程参数
    print("平面方程参数:")
    print(f"a: {params['a']}, b: {params['b']}, c: {params['c']}, d: {params['d']}")

    # -----------------------------------坐标变换---------------------------------------
    # # 从 .mat 文件中加载数据
    # mat = scipy.io.loadmat(r'C:\Users\lee\Desktop\项目\HHG\algorithm\algorithm\transfMtx.mat')
    # # 获取变量
    # transfMtx = mat['transfMtx']
    # #ld2ts
    # normal_vector_ld = np.array([params['a'],  params['b'],  params['c']])
    # normal_vector_ts = np.dot(transfMtx, np.append(normal_vector_ld, 1))
    # print("normal_vector_ts",normal_vector_ts)
    #
    # row_vector_ld = np.array(row_vector)
    # row_vector_ts = np.dot(transfMtx, np.append(row_vector_ld, 1))
    #
    # centroid_ld=centroid
    # centroid_ts = np.dot(transfMtx, np.append(centroid_ld, 1))
    # print('centroid_ts',centroid_ts)

    # -----------------------------------位姿计算---------------------------------------
    # 测试函数
    normal_vector = np.array([params['a'], params['b'], params['c']])
    normal_vector = ensure_direction(normal_vector)
    row_vector = np.array(row_vector)
    # 坐标变换后的
    # normal_vector = normal_vector_ts[:3]
    # row_vector = row_vector_ts[:3]

    yaw = yaw_pitch_row(normal_vector, projection_plane='yaw')
    print("偏航角(yaw)：", yaw)

    pitch = yaw_pitch_row(normal_vector, projection_plane='pitch')
    print("俯仰角(pitch)：", pitch)

    roll = yaw_pitch_row(row_vector, projection_plane='roll')
    print("横滚角(roll)：", roll)

    points = np.asarray(inlier_pcd.points)
    print(points.shape)
    boundarys_points = edge_estimate(points, vis=True)
    # 找角点
