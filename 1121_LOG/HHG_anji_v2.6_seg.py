# 使用强度区分  标定板
# 添加了mqtt功能，用以接收船只中部的姿态角信息（可搭配publish.py进行测试）

"""
    @ author: zhu
    整个代码实现：通过摄像头和雷达数据进行YOLO目标检测、点云处理、平面拟合、质心提取、以及使用MQTT通信功能将姿态信息发布到指定的主题。
    核心流程：
        1.【日志记录配置】：为代码执行配置日志记录，以便在文件中保存执行过程中的关键信息。
        2.【参数和矩阵初始化】：初始化设备的旋转矩阵、YOLO模型、DBSCAN聚类算法等必要参数。
        3.【YOLO目标检测】：加载YOLO模型，对图像数据进行目标检测。
        4.【ROS数据同步】：使用ROS节点同步获取来自点云和图像的话题信息，进行数据融合。
        5.【数据处理与点云操作】：通过聚类和点云处理来提取目标对象的质心、平面和角点等信息。
        6.【MQTT通信】：将姿态信息通过MQTT协议发送到指定的主题中。
        7.【坐标变换与位姿计算】：将目标的局部坐标转换为全局坐标系中的NEZ坐标。
"""

import os
import threading

import rospy
from sensor_msgs.msg import Image
import message_filters
from sensor_msgs.msg import PointCloud2
from sensor_msgs import point_cloud2
import cv2
import open3d as o3d
import numpy as np
import scipy
import sys
from pathlib import Path
import torch
import logging
import subscriber
import json
# import cupy as cp
import logging
import os
from logging.handlers import TimedRotatingFileHandler
from datetime import datetime, timedelta
from ultralytics import YOLO

mqtt_road = None

"""
    1.日志记录配置
    使用Python的logging库创建【日志记录器】，
    将运行过程中的关键信息保存到【指定路径的日志文件】中，便于调试和监控。
"""
# --------------------------------------------------------
# 配置日志记录器
logger = logging.getLogger('test')
logger.setLevel(logging.INFO)

# 创建文件处理器，使用追加模式
filename = os.path.join(r"/home/nvidia/yolov5-master/LOG")
file_handler = TimedRotatingFileHandler(filename=filename + '/HHG.log', when="MIDNIGHT", interval=1, backupCount=365)
file_handler.setLevel(logging.INFO)

# 创建日志格式
formatter = logging.Formatter('%(asctime)s.%(msecs)03d-%(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
file_handler.setFormatter(formatter)

# 将处理器添加到记录器
logger.addHandler(file_handler)
# --------------------------------------------------------


"""
    2.参数和矩阵初始化
    初始化了摄像头的内参矩阵K、旋转矩阵R、平移向量t以及两个外部文件中的变换矩阵。
    加载这些矩阵有助于将点云从雷达坐标系转换到相机坐标系，并完成数据融合。
"""

# --------------------------------------------------------
# RS速腾
R = np.array([[-0.188936, -0.981901, 0.0131859],
              [-0.0715861, 0.000379992, -0.997434],
              [0.979377, -0.189396, -0.0703623]])
t = np.array([-0.879042, 0.441192, 1.34659])
K = np.array([[1950.09, 0, 688.723],
              [0, 1936.29, 496.253],
              [0, 0, 1]])

axis_pcd = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1, origin=[0, 0, 0])
# 从 .mat 文件中加载数据
mat_4x4 = scipy.io.loadmat(r'/home/nvidia/yolov5-master/mat/tail/chuanwei4x4.mat')
mat_3x3 = scipy.io.loadmat(r'/home/nvidia/yolov5-master/mat/tail/chuanwei3x3.mat')
# --------------------------------------------------------

"""
    3.YOLO目标检测
    加载YOLO模型，并使用预训练的权重文件对摄像头图像进行目标检测。
    该模块还设置了模型的置信度阈值、NMS（非极大值抑制）阈值等超参数
"""
# --------------------------------------------------------
# YOLO检测相关内容
model_seg_path = r"C:\Users\admin\Desktop\yolov5-master\YOLO_segment.pt"

# 定义DBSCAN聚类的参数
eps = 0.3  # 邻域搜索半径
min_points = 50  # 最小邻域内点的数量

distance_threshold = 0.05  # 阈值用于确定点是否位于平面上
ransac_n = 3  # 用于拟合平面的随机样本点的数量
num_iterations = 1000  # RANSAC算法迭代次数

"""
    5.数据处理与点云操作
    Clustering函数：基于DBSCAN算法对点云进行聚类，用于过滤噪声点并提取目标点云簇。
"""


# 函数会找到拥有最多点的簇，并返回其对应的点云。
def Clustering(point_cloud, eps, min_points):
    # print("点云数量：", np.asarray(point_cloud.points).shape)
    # 进行聚类
    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
        labels = np.array(point_cloud.cluster_dbscan(eps=eps, min_points=min_points, print_progress=True))
    # labels = np.array(point_cloud.cluster_dbscan(eps=eps, min_points=min_points, print_progress=True))
    if len(labels) > 0:
        max_label = max(labels)
        min_label = min(labels)
        print(f"label:{min_label}~{max_label}")
        if max_label == -1:
            logger.warning("聚类：均为噪声点")
            return "error"
        MAX = 0  # 记录label中点最多的点的个数
        LABEL = 0  # 记录点最多的label的label数

        # 循环获取拥有最多点的label，并展示点数大于100的label
        for label in range(min_label + 1, max_label + 1):
            label_index = np.where(labels == label)
            label_pcd = point_cloud.select_by_index(np.array(label_index)[0])
            # print('label:', str(label), '点云数量：', len(label_pcd.points))
            if MAX < len(label_pcd.points):
                MAX = len(label_pcd.points)
                LABEL = label

        # 筛选拥有最多点的label，认定为目标人物点云
        label_index = np.where(labels == LABEL)
        person_part_pcd = point_cloud.select_by_index(np.array(label_index)[0])
        person_part_pcd.paint_uniform_color([0, 0, 1])
        return person_part_pcd


"""
    detect函数是YOLO检测的核心，将图像输入模型进行推理，输出目标检测框的坐标】类别和置信度。
"""


def detect(img):
    # 开始预测
    model.warmup(imgsz=(1, 3, *imgsz))  # warmup
    dt, seen = [0.0, 0.0, 0.0], 0

    # 对图片进行处理
    im0 = img
    # Padded resize
    im = letterbox(im0, imgsz, stride, auto=pt)[0]
    # Convert
    im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
    im = np.ascontiguousarray(im)
    t1 = time_sync()
    im = torch.from_numpy(im).to(device)
    im = im.half() if half else im.float()  # uint8 to fp16/32
    im /= 255  # 0 - 255 to 0.0 - 1.0
    if len(im.shape) == 3:
        im = im[None]  # expand for batch dim
    t2 = time_sync()
    dt[0] += t2 - t1

    # 预测
    pred = model(im, augment=augment, visualize=visualize)
    t3 = time_sync()
    dt[1] += t3 - t2

    # NMS
    pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
    dt[2] += time_sync() - t3

    # 用于存放结果
    detections = []

    # Process predictions
    for i, det in enumerate(pred):  # per image 每张图片
        seen += 1
        # im0 = im0s.copy()
        if len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()
            # Write results
            # 写入结果
            for *xyxy, conf, cls in reversed(det):
                xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4))).view(-1).tolist()
                xywh = [round(x) for x in xywh]
                xywh = [xywh[0] - xywh[2] // 2, xywh[1] - xywh[3] // 2, xywh[2],
                        xywh[3]]  # 检测到目标位置，格式：（left，top，w，h）

                cls = names[int(cls)]
                conf = float(conf)
                detections.append({'class': cls, 'conf': conf, 'position': xywh})
    # 输出结果
    # for i in detections:
    # print(i)

    # 推测的时间
    LOGGER.info(f'({t3 - t2:.3f}s)')
    return detections


# 功能: 该函数根据法向量在不同投影平面上的投影，计算出偏航、俯仰或滚转角度，并以度数形式返回。
def yaw_pitch_row(normal_vector, projection_plane='yaw'):
    if projection_plane == 'yaw':
        # xoy 平面的法向量
        plane_normal = np.array([0, 0, 1])
        axis_vector = np.array([1, 0, 0])  # x 轴
    elif projection_plane == 'pitch':
        # yoz 平面的法向量
        plane_normal = np.array([1, 0, 0])
        axis_vector = np.array([0, -1, 0])  # y 轴
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

    # print(projection_vector, cos_angle)
    angle = np.arccos(cos_angle)

    # 确定角度的符号
    if projection_plane == 'yaw':
        # 如果投影向量在 y 轴左侧（即 x 轴正方向），角度为正；否则为负
        # if projection_vector[1] <= 0:
        return np.degrees(angle)
    # else:
    # return -np.degrees(angle)
    elif projection_plane == 'pitch':
        # 如果投影向量在 y 轴以上（即 z 轴正方向），角度为正；否则为负
        # if projection_vector[2] >= 0:
        return np.degrees(angle)
    # else:
    #     return -np.degrees(angle)
    elif projection_plane == 'roll':
        # 如果投影向量在 x 轴以上（即 z 轴正方向），角度为正；否则为负
        if projection_vector[2] >= 0:
            return np.degrees(angle)
        else:
            return -np.degrees(angle)


"""
    call_back函数
    这是一个典型的ROS回调函数，在这个代码中，用于处理ROS话题的数据。
    
    1.【回调函数在ros中的作用】：
        回调函数是响应订阅的数据到达时执行的函数。
        当一个 ROS 节点订阅了某个话题后，每当新消息发布到这个话题上，
        ROS 就会调用订阅者注册的回调函数来处理数据。这样可以实现数据的实时处理和系统的响应。
        
    2.【call_back 函数在该代码中的作用】：
        在这个代码中，call_back 函数负责同步处理来自激光雷达的点云数据 (ros_pcd) 和来自相机的图像数据 (ros_img)。
        工作流程：
        (1)数据读取和过滤
            call_back 函数首先从订阅的 /middel/rslidar_points 和 /image1 话题中读取点云和图像数据，
            并对点云数据进行初步过滤。代码中通过 message_filters 的
             ApproximateTimeSynchronizer 进行数据同步，以确保在处理时点云和图像数据是接近同一时间的。
        (2)点云数据处理
            从 ros_pcd 中读取到的点云数据被转换为 NumPy 数组，
            并对点云坐标进行过滤。代码中对点云的纵深 (y) 方向进行筛选，
            以减少无关数据的干扰
        (3)图像数据处理
            call_back 函数还从 ros_img 中提取图像数据并转换为 OpenCV 格式，以便后续通过 YOLO 进行目标检测。
        (4)YOLO目标检测
            经过处理的图像数据被送入 YOLO 模型进行目标检测。
            检测结果包括物体类别、置信度和位置信息，用于后续的点云与图像的融合。
        (5)数据融合和平面拟合
            根据检测到的目标框，对点云和图像进行融合。
            代码中通过聚类、平面拟合和质心计算等方法，从点云数据中提取出重要的几何特征。
            这些特征随后用于计算目标物体的位姿信息（如偏航角、俯仰角）。
        (6)MQTT数据发布
            最后，call_back 函数通过 MQTT 将计算得到的位姿和坐标数据发布出去，以便在其他节点或系统中使用。
            
    3.总结：call_back函数的核心流程
        1.同步获取点云和图像数据
        2.对点云数据进行过滤并处理
        3.对图像数据进行YOLO目标检测
        4.融合图像点云数据，并提取目标位置和姿态信息
        5.使用MQTT将处理结果发布出去
"""

def call_back(ros_pcd, ros_img):
    try:
        # ----------------------------------------原始点云数据获取---------------------------------------------------
        points = []
        for msg in point_cloud2.read_points_list(ros_pcd,
                                                 field_names=('x', 'y', 'z', 'intensity'),
                                                 skip_nans=True):
            points.append([msg[0], msg[1], msg[2], msg[3]])
        # print(points.shape)
        points = np.array(points)
        print(points.shape)
        point_cloud = o3d.geometry.PointCloud()  # point_cloud为空的点云对象
        after_filter = np.logical_and.reduce(
            [
                # points[:, 0] >= 0, points[:, 0] <= 20,  # 水平方向
                points[:, 1] >= -40, points[:, 1] <= 2,  # 纵深
                # points[:, 2] >= -10.5, points[:, 2] <= 1, # 垂直方向
            ]
        )

        point_cloud.points = o3d.utility.Vector3dVector(points[after_filter][:, 0:3])
        intensities_filter = points[after_filter][:, 3]
        intensities_filter = intensities_filter.reshape(points[after_filter].shape[0],1)

        image = np.frombuffer(ros_img.data, dtype=np.uint8).reshape(720, 1280, -1)  # rgb格式
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # o3d.visualization.draw_geometries([point_cloud, axis_pcd], window_name=f"原始点云图_{index}")
        print("符合范围筛选条件的点云数量: ", np.asarray(point_cloud.points).shape)
        # ---------------------------------------yolo语义分割-------------------------------------------------

        model = YOLO(model_seg_path)
        results = model.predict(image)
        masksxy = results[0].masks.xy

        # 创建一个空的掩膜，数据类型改为 uint8
        mask_bool = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)

        # 将边界点转换为填充掩膜
        for mask in masksxy:
            # 将点转换为整数并填充
            points = mask.astype(np.int32)
            cv2.fillPoly(mask_bool, [points], 1)  # 使用 1 来填充

        # 转换回布尔类型（可选）
        mask_bool = mask_bool.astype(bool)


        # ----------------------------------------数据融合-----------------------------------------------------

        cloud = np.asarray(point_cloud.points)
        colored_points = []
        for point in cloud:
            # 点云转换到相机坐标系
            pL = np.array([point[0], point[1], point[2], 1.0])
            pC = R @ pL[:3] + t

            # 投影到图像平面
            q = K @ pC
            u = int(q[0] / q[2])
            v = int(q[1] / q[2])

            # 检查是否在图像范围内且在掩膜内
            if 0 <= u < image.shape[1] and 0 <= v < image.shape[0] and mask_bool[v, u]:
                color = image[v, u]  # 提取图像中的颜色信息
                colored_points.append(
                    [point[0], point[1], point[2], color[2] / 255.0, color[1] / 255.0, color[0] / 255.0]  # RGB归一化
                )
        # 创建一个新的点云对象 colored_cloud 用于存储彩色点信息
        colored_cloud = o3d.geometry.PointCloud()
        print("打印提取到的彩色点数量: ", len(colored_points))  # 打印提取到的彩色点数量
        # 将带有颜色信息的点和对应的颜色设置到 colored_cloud 对象中
        colored_cloud.points = o3d.utility.Vector3dVector(np.array(colored_points)[:, :3])  # 提取位置坐标
        colored_cloud.colors = o3d.utility.Vector3dVector(np.array(colored_points)[:, 3:])  # 提取颜色信息
        # 可视化彩色点云
        # o3d.visualization.draw_geometries([colored_cloud, point_cloud], window_name="融合后彩色点云")

        # ---------------------------------------拟合-----------------------------------------------------

        """
            5.数据处理与点云操作
            平面拟合：通过RANSAC算法对点云进行平面拟合，提取目标平面
        """
        plane_model, inliers = colored_cloud.segment_plane(distance_threshold=0.05, ransac_n=3,
                                                           num_iterations=100000)
        # print("第一次平面拟合结束时间: ", time.time())  # 平面拟合的耗时
        [a, b, c, d] = plane_model  # 解包平面模型参数
        print("语义分割-面方程参数: ")
        print(f"语义分割-拟合得到面方程及参数: {a:.6f}x + {b:.6f}y + {c:.6f}z + {d:.6f} = 0")

        # --------------------------------------按平面方程过滤原始点云-----------------------------------------------------
        # 5. 过滤并提取船尾平面点云
        raw_points = np.asarray(point_cloud.points)
        tail_point_cloud = o3d.geometry.PointCloud()  # 创建新的点云对象 tail_point_cloud,用于存储过滤后的点云数据
        # 条件过滤: 以提取位于船尾平面附近的点
        after_filter = np.logical_and.reduce([
            a * raw_points[:, 0] + b * raw_points[:, 1] + c * raw_points[:, 2] + d >= -0.5,
            a * raw_points[:, 0] + b * raw_points[:, 1] + c * raw_points[:, 2] + d <= 0.5
        ])
        tail_point_cloud.points = o3d.utility.Vector3dVector(raw_points[after_filter])
        # tail_point_cloud.paint_uniform_color([1, 0, 0])  # 船尾面为红色
        # o3d.visualization.draw_geometries([tail_point_cloud], window_name=f"船尾平面点云_{index}")

        # ----------------------求解标定板质心-------------------------------
        raw_pcd_intensity_points = np.hstack((raw_points, intensities_filter))  # intensities_filter是通过范围过滤后的

        tail_pcd_intensity_points = raw_pcd_intensity_points[after_filter]  # 根据bool筛选符合的行
        tail_pcd_intensity_points = np.array(tail_pcd_intensity_points, dtype=float)
        after_filter_intensity = np.logical_and.reduce(
            [
                tail_pcd_intensity_points[:, 3] > 180  # 强度值大于 180
            ]
        )
        flag = tail_pcd_intensity_points[after_filter_intensity].shape[0]  # 返回符合条件的行数flag是数量

        if flag > 0:
            # if flag != 12:
            #     logger.warning(f"标定板筛选不等于12个点 flag = {flag}")
            ROI_pcd = o3d.geometry.PointCloud()  # 通过强度筛选的
            ROI_pcd.points = o3d.utility.Vector3dVector(tail_pcd_intensity_points[after_filter_intensity][:, :3])
            ROI_pcd.paint_uniform_color([0, 1, 0])  # 第二步分割后的点云涂上黄色

            """
                5.数据处理与点云操作
                质心提取：通过强度值筛选点云中的高强度点，并计算质心位置。
            """
            centroid = np.mean(tail_pcd_intensity_points[after_filter_intensity][:, :3], axis=0)

            centroid_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.04)
            centroid_sphere.translate(centroid)
            centroid_sphere.paint_uniform_color([1, 0, 0])  # 质心上色为红色
            logger.info(f"质心坐标：{centroid}")
            print("质心坐标：", centroid)
            # o3d.visualization.draw_geometries([centroid_sphere, ROI_pcd, tail_point_cloud], window_name=f"质心_{index}")
        else:
            print("没有放置标定板或标定板反射效果不好")
            logger.warning("没有放置标定板或标定板反射效果不好")
        # ----------------------特征点提取-----------------------------------

        corner_pcd = Clustering(tail_point_cloud, eps=0.5, min_points=20)

        # o3d.visualization.draw_geometries([corner_pcd, axis_pcd], window_name=f"聚类1_{index}")

        tpcd_points = np.asarray(corner_pcd.points)
        min_val = corner_pcd.get_min_bound()
        max_val = corner_pcd.get_max_bound()
        print(min_val[0], min_val[0] + (max_val[0] - min_val[0]))
        # 提取ROI 提取点云中指定范围内的感兴趣区域，用于进一步聚类或特征提取。
        tail_ROI_pcd = o3d.geometry.PointCloud()
        after_filter = np.logical_and.reduce(
            [
                tpcd_points[:, 0] >= min_val[0], tpcd_points[:, 0] <= min_val[0] + (max_val[0] - min_val[0]) / 10
            ]
        )
        tail_ROI_pcd.points = o3d.utility.Vector3dVector(
            tpcd_points[after_filter]
        )
        # o3d.visualization.draw_geometries([tail_ROI_pcd])

        corner_pcd = tail_ROI_pcd
        # o3d.visualization.draw_geometries([corner_pcd, axis_pcd], window_name=f"聚类2_{index}")
        #
        # 获取点云的点坐标
        corner_points = np.asarray(corner_pcd.points)

        # --------------------------找到最左上角的角点-----------------------------
        # 方案一
        # 最小x坐标
        # min_x_index = np.argmin(corner_points[:, 0])
        # left_point_index = np.where((corner_points[:, 0] == corner_points[min_x_index, 0]))
        # # print(left_point_index)
        # left_point = corner_points[left_point_index]
        # # print(left_point)
        # # 最大z坐标
        # max_z_index = np.argmax(left_point[:, 2])
        # top_point_index = np.where((left_point[:, 2] == left_point[max_z_index, 2]))
        # # print(top_point_index)
        # left_top_point = left_point[top_point_index]

        # # 方案二
        # # 取反x坐标
        neg_x = -corner_points[:, 0]
        # 计算加权和(加权因子可调)
        weighted_sum = 0.4 * neg_x + 0.6 * corner_points[:, 2]
        # 找到和最大的点的索引
        max_index = np.argmax(weighted_sum)
        # 提取和最大的点
        left_top_point = corner_points[max_index]
        left_top_point = [left_top_point]

        corner = o3d.geometry.PointCloud()
        corner.points = o3d.utility.Vector3dVector(left_top_point)
        corner.paint_uniform_color([1, 0, 0])
        print("角点点云坐标：", np.asarray(corner.points))
        logger.info(f"角点点云坐标：{np.asarray(corner.points)}")

        centroid_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.1)
        centroid_sphere.translate(left_top_point[0])
        centroid_sphere.paint_uniform_color([1, 0, 0])  # 质心上色为红色

        # o3d.visualization.draw_geometries([centroid_sphere, point_cloud], window_name=f"角点提取_{index}")

        # -----------------------------------坐标变换---------------------------------------

        transfMtx = mat_3x3['transfMtx']
        # print(transfMtx)
        transfMtx = np.array([[ 0.1265,    -0.99054,   0.053148,   0],
                              [ 0.99192,    0.12577,  -0.016816,   0],
                              [-0.0099724,  0.054845, -0.99844,    0],
                              [ 0,          0,         0,          1]])
        # ld2ts
        normal_vector_ld = np.array([a, b, c])
        # print(np.append(normal_vector_ld, 1).shape)
        normal_vector_ts = np.dot(transfMtx, np.append(normal_vector_ld, 1))
        # print("ts", normal_vector_ts)
        # print(normal_vector_ts.shape)
        corner_ld = np.asarray(corner.points)
        transfMtx = mat_4x4['transfMtx']
        # transfMtx = np.array([[ 0.99635,     0.09061,    0.032129,      540.39],
        #                      [ -0.10234,     0.99929,    0.059005,       -2371],
        #                      [  0.042613,    0.050914,     -1.0003,     1.8164],
        #                      [  0,           0,           0,       1]])
        corner_ts = np.dot(transfMtx, np.append(corner_ld, 1))
        corner_ts = [corner_ts[0], -corner_ts[1], -corner_ts[2]]
        print(f'角点NEZ坐标：{corner_ts}')
        logger.info(f'角点NEZ坐标：{corner_ts}')
        centroid_ld = centroid
        centroid_ts = np.dot(transfMtx,np.append(centroid_ld,1))
        centroid_ts = [centroid_ts[0], -centroid_ts[1], -centroid_ts[2]]
        print(f"质心NEZ坐标 : {centroid_ts}")
        logger.info(f"质心NEZ坐标 : {centroid_ts}")
        # -----------------------------------位姿计算---------------------------------------

        if np.dot(normal_vector_ts[0:3], [0, -1, 0]) < 0:
            normal_vector = -normal_vector_ts[:3]
        else:
            normal_vector = normal_vector_ts[:3]
        # print(normal_vector)

        """
            7.坐标变换与位姿计算
            该模块通过变换矩阵将局部坐标转化为NEZ坐标系，
            并计算偏航角、俯仰角等姿态角度，
            最后将信息以JSON格式发送到MQTT主题。
        """

        yaw = yaw_pitch_row(normal_vector, projection_plane='yaw')
        print("偏航角(yaw)：", yaw)
        logger.info(f"偏航角(yaw)：{yaw}")
        pitch = yaw_pitch_row(normal_vector, projection_plane='pitch')
        print("俯仰角(pitch)：", pitch)
        logger.info(f"俯仰角(pitch)：{pitch}")

        data = subscriber.received_message
        roll = data.payload.decode('UTF-8')
        print(f"多线程接收到的信息: {roll}")
        # 发消息
        mqtt_json_data = '{"Sys_state":"0","Gps_state":"0",' \
                         '"Pitch":"0","Roll":"0","Yaw":"0",' \
                         '"Ve":"0","Vn":"0","Vt":"0",' \
                         '"Lon":"0","Lat":"0","Hei":"0",' \
                         '"Pn":"0","Pe":"0","Pu":"0",' \
                         '"Px":"0","Py":"0","Pz":"0",' \
                         '"DeviceId":"98","Qoe":"0"}'
        mqtt_data = json.loads(mqtt_json_data)
        # print(f'mqtt的json发送前数据{mqtt_data},类型为{type(mqtt_data)}')
        mqtt_data["Pitch"] = pitch
        mqtt_data["Roll"] = roll
        mqtt_data["Yaw"] = yaw
        mqtt_data["Pn"] = centroid_ts[0]
        mqtt_data["Pe"] = centroid_ts[1]
        mqtt_data["Pu"] = centroid_ts[2]
        # print(f'mqtt的json要发送数据{mqtt_data}')
        sendMessage(mqtt_data)

    except Exception as e:
        mqtt_json_data = '{"Sys_state":"0","Gps_state":"0",' \
                         '"Pitch":"0","Roll":"0","Yaw":"0",' \
                         '"Ve":"0","Vn":"0","Vt":"0",' \
                         '"Lon":"0","Lat":"0","Hei":"0",' \
                         '"Pn":"0","Pe":"0","Pu":"0",' \
                         '"Px":"0","Py":"0","Pz":"0",' \
                         '"DeviceId":"98","Qoe":"0"}'
        mqtt_data = json.loads(mqtt_json_data)
        sendMessage(mqtt_data)
        print(e)
        logger.exception(e)
        pass


def mqtt_thread_function():
    global mqtt_road
    mqtt_thread = subscriber.MqttRoad('10.60.127.145', 5195, 600)
    mqtt_road = mqtt_thread
    mqtt_thread.start()


def sendMessage(message):
    if mqtt_road:
        message = json.dumps(message)
        # mqtt_road.publish_message('topic/ship', message)
        status = mqtt_road.publish_message('topic/ship', message)
        if status == 0:
            print(f"success")
        else:
            print(f"fail")
    else:
        print("MQTT 客户端未初始化。")


if __name__ == '__main__':

    """
        6.MQTT通信
        使用MQTT协议将目标检测得到的姿态信息发布到指定的主题。
        mqtt_thread_function用于启动MQTT线程，并接受或发布数据。
    """

    # --------------------------------------------------------------
    # 为mqtt创建一个线程
    mqtt_thread = threading.Thread(target=mqtt_thread_function)
    mqtt_thread.start()
    # --------------------------------------------------------------

    rospy.init_node('cloud_img_process', anonymous=True)
    print('初始化')

    """
        4.ROS数据同步
        使用ROS的[message_filters]同步订阅点云和图像数据，通过call_back函数进行处理。
        ApproximaterTimeSynchronizer保证两个不同源的数据同步到同一时刻。
    """
    # --------------------------------------------------------
    ros_pcd = message_filters.Subscriber("/middle/rslidar_points", PointCloud2, queue_size=1)
    ros_img = message_filters.Subscriber("/image1", Image, queue_size=1)
    ts = message_filters.ApproximateTimeSynchronizer([ros_pcd, ros_img], 10, 0.1, allow_headerless=True)
    ts.registerCallback(call_back)
    # --------------------------------------------------------

    # 循环来源
    rospy.spin()  # spin() simply keeps python from exiting until this node is stopped
