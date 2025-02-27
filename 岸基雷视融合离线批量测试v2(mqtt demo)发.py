# 将MSAC转为open3d的RANSAC
import json
import time

import cv2
import open3d as o3d
import numpy as np
import scipy
import sys
from pathlib import Path
import torch
# import cupy as cp
import logging
import os
import pandas as pd
import subscriber
import threading

mqtt_road=None
# 配置日志记录器
logger = logging.getLogger('test')
logger.setLevel(logging.INFO)

# 创建控制台处理器
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

# 创建文件处理器，使用追加模式
file_handler = logging.FileHandler('../HHG.log', mode='a')
file_handler.setLevel(logging.INFO)

# 创建日志格式
formatter = logging.Formatter('%(asctime)s.%(msecs)03d- %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
# console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

# 将处理器添加到记录器
# logger.addHandler(console_handler)
logger.addHandler(file_handler)

# K = np.array([[1900.2, 0, 688.723],
#               [0, 1911.36, 496.253],
#               [0, 0, 1]])
# # 旋转矩阵
# R = np.array([[0.973976, -0.226649, 0.000702814],
#               [0.00405423, 0.0143216, -0.999889],
#               [0.226614, 0.973871, 0.0148678]])
# # 平移矩阵
# t = np.array([-0.137048, -0.00500911, 0.0110638])
#
# K = np.array([[4201.31, 0, 1148.87],
#               [0, 4190.06, 836.582],
#               [0, 0, 1]])
# # 旋转矩阵
# R = np.array([[0.0944619, -0.994604, -0.0428947],
#               [-0.0326906, 0.0399651, -0.998666],
#               [0.994992, 0.0957382, -0.028739]])
# 平移矩阵
# t = np.array([-1.52616, -0.0140314, 9.63334])


K = np.array([[1950.09, 0, 688.723],
              [0, 1936.29, 496.253],
              [0, 0, 1]])

# # 旋转矩阵
R = np.array([[0.975092, -0.221687, 0.00713732],
              [0.00688296, -0.00191993, -0.999974],
              [0.221695, 0.975116, -0.000346246]])

# 平移矩阵X
t = np.array([-0.424563, 0.535128, 1.08017])

# 点云坐标系对象
axis_pcd = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1, origin=[0, 0, 0])
# 从 .mat 文件中加载数据
mat_4x4 = scipy.io.loadmat(r'transfMtx_HHG.mat')
mat_3x3 = scipy.io.loadmat(r'RotfMtx.mat')
# YOLO检测相关内容
# 初始化目录
FILE = Path(__file__).resolve()

ROOT = FILE.parents[0]  # 定义YOLOv5的根目录
print(f'{ROOT}')
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # 将YOLOv5的根目录添加到环境变量中（程序结束后删除）
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.general import (LOGGER, check_img_size, non_max_suppression, scale_boxes, xyxy2xywh)
from utils.torch_utils import select_device, time_sync
from utils.augmentations import letterbox

weights = r'HHG0822.pt'  # 权重文件地址   .pt文件
data = ROOT / 'data/HHGdata.yaml'  # 标签文件地址   .yaml文件

imgsz = (640, 640)  # 输入图片的大小 默认640(pixels)
conf_thres = 0.5  # object置信度阈值 默认0.25  用在nms中
iou_thres = 0.45  # 做nms的iou阈值 默认0.45   用在nms中
max_det = 1000  # 每张图片最多的目标数量  用在nms中
device = 'cpu'  # 设置代码执行的设备 cuda device, i.e. 0 or 0,1,2,3 or cpu
classes = None  # 在nms中是否是只保留某些特定的类 默认是None 就是所有类只要满足条件都可以保留 --class 0, or --class 0 2 3
agnostic_nms = False  # 进行nms是否也除去不同类别之间的框 默认False
augment = False  # 预测是否也要采用数据增强 TTA 默认False
visualize = False  # 特征图可视化 默认FALSE
half = True  # 是否使用半精度 Float16 推理 可以缩短推理时间 但是默认是False
dnn = False  # 使用OpenCV DNN进行ONNX推理

# 获取设备
device = select_device(device)

# 载入模型
model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data)
stride, names, pt, jit, onnx, engine = model.stride, model.names, model.pt, model.jit, model.onnx, model.engine
imgsz = check_img_size(imgsz, s=stride)  # 检查图片尺寸

# Half
# 使用半精度 Float16 推理
half &= (pt or jit or onnx or engine) and device.type != 'cpu'  # FP16 supported on limited backends with CUDA
if pt or jit:
    model.model.half() if half else model.model.float()

# 定义DBSCAN聚类的参数
eps = 0.3  # 邻域搜索半径
min_points = 50  # 最小邻域内点的数量

distance_threshold = 0.05  # 阈值用于确定点是否位于平面上
ransac_n = 3  # 用于拟合平面的随机样本点的数量
num_iterations = 1000  # RANSAC算法迭代次数


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


def detect(img):
    # 开始预测
    # warmup模型预热是为了在实际推理前进行一次快速的前向传播，以便初始化模型的内部状态和优化性能
    # *imgsz实际会被解释为(1(batch),3(维度),长,宽)
    model.warmup(imgsz=(1, 3, *imgsz))  # warmup
    # 初始化一个列表 dt 用于记录不同阶段的时间消耗，seen 用于计数处理过的图像数量
    dt, seen = [0.0, 0.0, 0.0], 0

    # 对图片进行处理
    im0 = img
    # Padded resize 使用 letterbox 函数将输入图像 im0 按照给定的尺寸 imgsz 和步幅 stride 进行填充缩放，以保持图像的长宽比
    im = letterbox(im0, imgsz, stride, auto=pt)[0]
    # Convert
    im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
    # 确保图像在内存中是连续的
    im = np.ascontiguousarray(im)
    #使用 torch.from_numpy 将 NumPy 数组转换为 PyTorch 张量，并将其移动到指定的设备（CPU 或 GPU）。
    #根据是否启用半精度，将图像数据类型转换为 Float16 或 Float32。
    # 将像素值从 0-255 标准化到 0.0-1.0。
    t1 = time_sync()
    im = torch.from_numpy(im).to(device)
    im = im.half() if half else im.float()  # uint8 to fp16/32
    im /= 255  # 0 - 255 to 0.0 - 1.0
    # 如果输入图像只有三个维度（即单张图像），则通过添加一个新的维度来扩展为批量维度，以便模型可以处理
    if len(im.shape) == 3:
        im = im[None]  # expand for batch dim
    # 记录图像处理阶段所消耗的时间
    t2 = time_sync()
    dt[0] += t2 - t1

    # 预测
    pred = model(im, augment=augment, visualize=visualize)
    # 记录模型预测阶段所消耗的时间
    t3 = time_sync()
    dt[1] += t3 - t2

    # NMS 对模型预测的结果应用非极大值抑制，以消除冗余的检测框，保留置信度最高的框
    pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
    dt[2] += time_sync() - t3

    # 用于存放结果
    detections = []

    # Process predictions
    # enumerate是迭代器，遍历的时候，i是index，而det是遍历的内容，det是每个检测框
    for i, det in enumerate(pred):  # per image 每张图片
        seen += 1
        # im0 = im0s.copy()
        if len(det):
            # 这段代码的主要功能是将检测到的边界框从模型的输入尺寸转换为原始图像的尺寸，
            # 以便在原始图像上正确绘制这些边界框。这是目标检测任务中的常见步骤，确保检测结果与原始图像对齐
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


# def MSAC_GPU(data):
#     data = cp.asarray(data)  # 将数据转为GPU上的数组*/ 11
#     number = data.shape[1]
#     iter = 5000
#     sigma = 0.005
#     preF = cp.inf
#
#     bestplane = None
#
#     for _ in range(iter):
#         idx = cp.random.choice(number, 3, replace=False)
#         sample = data[:, idx]
#
#         p1 = sample[:, 0]
#         p2 = sample[:, 1]
#         p3 = sample[:, 2]
#
#         v1 = p2 - p1
#         v2 = p3 - p1
#
#         if cp.linalg.norm(cp.cross(v1, v2)) < 1e-6:
#             continue
#
#         nv = cp.cross(v1, v2)
#         nv = nv / cp.linalg.norm(nv)
#         nv = cp.append(nv, -cp.dot(nv, p1))
#
#         if nv[0] < 0 and abs(nv[0]) > 0.01:
#             nv = -nv
#
#         mask = cp.abs(cp.dot(nv, cp.vstack((data, cp.ones((1, data.shape[1]))))))
#
#         F1 = cp.sum(mask[mask < sigma])
#         F2 = cp.sum(mask > sigma) * 0.01
#         F = F1 + F2
#
#         if F < preF:
#             preF = F
#             bestplane = nv
#
#     mask = cp.abs(cp.dot(bestplane, cp.vstack((data, cp.ones((1, data.shape[1])))))) < sigma
#     inliers = data[:, mask]
#
#     para = {
#         'a': bestplane[0].get(),
#         'b': bestplane[1].get(),
#         'c': bestplane[2].get(),
#         'd': bestplane[3].get(),
#     }
#
#     return para, inliers.get()


def ensure_direction(vector, reference=np.array([0, -1, 0])):
    # 规范化向量
    vector_normalized = vector / np.linalg.norm(vector)
    reference_normalized = reference / np.linalg.norm(reference)
    # 如果点积为负，反转 vector 的方向
    if np.dot(vector_normalized, reference_normalized) < 0:
        vector_normalized = -vector_normalized
    return vector_normalized


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


def call_back(pcd, img, index, pcd_file_path, image_file_path, data,mqtt_thread):
    try:  # ----------------------------------------原始数据获取---------------------------------------------------
        # 记录原始点云数据
        points = np.asarray(pcd.points)

        point_cloud = o3d.geometry.PointCloud()
        after_filter = np.logical_and.reduce(
            [
                points[:, 0] >= 0, points[:, 0] <= 40,  # 水平方向过滤
                # points[:, 1] >= 2, points[:, 1] <= 40,    # 纵深
                # points[:, 2] >= -10.5, points[:, 2] <= 1, # 垂直方向
            ]
        )
        point_cloud.points = o3d.utility.Vector3dVector(
            points[after_filter]
        )
        # o3d.visualization.draw_geometries([point_cloud, axis_pcd])
        print("过滤后点云数量：", np.asarray(point_cloud.points).shape)

        # ---------------------------------------yolo识别定位-------------------------------------------------
        detections = detect(image)
        print("识别结果：", detections)
        logger.info(f"识别结果：{detections}")
        i = 0
        objects = []
        for detection in detections:
            obj = {'class': detection['class']}
            real_coordinate = (detection['position'][0], detection['position'][0] + detection['position'][2],
                               detection['position'][1], detection['position'][1] + detection['position'][3])
            obj['real_coordinate'] = real_coordinate
            i = i + 1
            objects.append(obj)
            print(detection['position'][0], detection['position'][2], detection['position'][1],
                  detection['position'][3])
            print(real_coordinate)
        # print(objects)

        if len(objects) > 0:
            # ----------------------------------------数据融合-----------------------------------------------------
            # 图像与雷达 数据层融合
            cloud = np.asarray(point_cloud.points)
            colored_points = []
            for point in cloud:
                # 点云转换到相机坐标系
                pL = np.array([point[0], point[1], point[2], 1.0])
                pC = R @ pL[:3] + t

                # 投影到图像平面
                q = K @ pC
                u = int(q[0] / q[2])  #像素坐标（u,v）
                v = int(q[1] / q[2])

                # 检查是否在图像范围内
                if (obj['real_coordinate'][0] <= u <= obj['real_coordinate'][1]) and \
                        (obj['real_coordinate'][2] <= v <= obj['real_coordinate'][3]):
                    color = image[v, u]  # 提取图像中的颜色信息
                    colored_points.append(
                        [point[0], point[1], point[2], color[2] / 255.0, color[1] / 255.0, color[0] / 255.0])  # RGB归一化
            colored_cloud = o3d.geometry.PointCloud()
            print(len(colored_points))
            if len(colored_points) == 0:
                logger.warning("yolo框内无点云")
                return
                # print()

            #拟合平面
            colored_cloud.points = o3d.utility.Vector3dVector(np.array(colored_points)[:, :3])
            colored_cloud.colors = o3d.utility.Vector3dVector(np.array(colored_points)[:, 3:])
            # o3d.io.write_point_cloud("yolo_color.pcd", colored_cloud)
            # o3d.visualization.draw_geometries([colored_cloud, point_cloud], window_name="融合后彩色点云")

            # ---------------------------------------拟合-----------------------------------------------------

            print(time.time())
            plane_model, inliers = colored_cloud.segment_plane(distance_threshold=0.005, ransac_n=3,
                                                               num_iterations=10000)
            print(time.time())
            [a, b, c, d] = plane_model
            print("YOLO识别面方程参数:")
            print(f"Plane equation: {a:.6f}x + {b:.6f}y + {c:.6f}z + {d:.6f} = 0")

            # ----------------------按YOLO识别结果过滤原始点云-----------------------------------------------------
            # 5. 过滤并提取船尾平面点云
            raw_points = np.asarray(point_cloud.points)
            tail_point_cloud = o3d.geometry.PointCloud()
            after_filter = np.logical_and.reduce([
                a * raw_points[:, 0] + b * raw_points[:, 1] + c * raw_points[:, 2] + d >= -0.1,
                a * raw_points[:, 0] + b * raw_points[:, 1] + c * raw_points[:, 2] + d <= 0.1
            ])
            tail_point_cloud.points = o3d.utility.Vector3dVector(raw_points[after_filter])
            tail_point_cloud.paint_uniform_color([1, 0, 0])
            # o3d.visualization.draw_geometries([tail_point_cloud], window_name=f"船尾平面点云_{index}")
            # o3d.io.write_point_cloud("tail.pcd", tail_point_cloud)

            # ----------------------按YOLO识别结果过滤原始点云-----------------------------------------------------

            print(time.time())
            # 6. 再次拟合船尾平面
            plane_model, inliers = tail_point_cloud.segment_plane(distance_threshold=0.005, ransac_n=3,
                                                                  num_iterations=100000)
            print(time.time())
            [a, b, c, d] = plane_model
            print("船尾面平面方程参数:")
            print(f"Plane equation: {a:.6f}x + {b:.6f}y + {c:.6f}z + {d:.6f} = 0")

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
            # o3d.io.write_point_cloud("zzz.pcd", tail_ROI_pcd)

            corner_pcd = tail_ROI_pcd
            # o3d.visualization.draw_geometries([corner_pcd, axis_pcd], window_name=f"聚类2_{index}")
            #
            # 获取点云的点坐标
            corner_points = np.asarray(corner_pcd.points)
            # # 是因为 json.dump 不能直接序列化 NumPy 的 ndarray 对象。你需要将其转换为列表格式。
            # corner_points_list = corner_points.tolist()
            # try:
            #     with open('data.json', 'w') as f:
            #         print(corner_points)
            #         print('将数据记录下来')
            #         json.dump(corner_points_list, f)
            # except Exception as e:
            #     print(f'发生错误: {e}')

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

            #-------------------------------------------------------

            # # # 取反x坐标
            neg_x = -corner_points[:, 0]
            # # 计算加权和(加权因子可调)
            weighted_sum = 0.4 * neg_x + 0.6 * corner_points[:, 2]
            # weighted_sum = neg_x + corner_points[:, 2]
            # # 找到和最大的点的索引
            max_index = np.argmax(weighted_sum)
            # 提取和最大的点
            left_top_point = corner_points[max_index]
            left_top_point = [left_top_point]

            print("left_top_point:", left_top_point)

            corner = o3d.geometry.PointCloud()
            corner.points = o3d.utility.Vector3dVector(left_top_point)
            corner.paint_uniform_color([1, 0, 0])
            # corner.points2 = o3d.utility.Vector3dVector(left_top_point2)
            # corner.paint_uniform_color([1, 0, 0])
            print("角点点云坐标：", np.asarray(corner.points))
            logger.info(f"角点点云坐标：{np.asarray(corner.points)}")

            centroid_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.1)
            centroid_sphere.translate(left_top_point[0])
            centroid_sphere.paint_uniform_color([1, 0, 0])  # 质心上色为红色

            # o3d.visualization.draw_geometries([centroid_sphere, point_cloud], window_name=f"角点提取_{index}")

            # -----------------------------------坐标变换---------------------------------------

            transfMtx = mat_3x3['transfMtx']
            # print(transfMtx)
            # transfMtx = np.array([[ 0.99635,     0.09061,    0.032129,      0],
            #                      [ -0.10234,     0.99929,    0.059005,       0],
            #                      [  0.042613,    0.050914,     -1.0003,       0],
            #                      [  0,           0,           0,       1]])
            # #ld2ts
            normal_vector_ld = np.array([a, b, c])
            # normal_vector_ld = np.array([tail_params['a'], tail_params['b'], tail_params['c']])
            # print(np.append(normal_vector_ld, 1).shape)
            normal_vector_ts = np.dot(transfMtx, np.append(normal_vector_ld, 1))
            # print("ts", normal_vector_ts)
            # print(normal_vector_ts.shape)
            corner_ld = np.asarray(corner.points)
            print('做变换的点', corner_ld)
            transfMtx = mat_4x4['transfMtx']
            print('transfMtx矩阵', transfMtx)
            # transfMtx = np.array([[ 0.99635,     0.09061,    0.032129,      540.39],
            #                      [ -0.10234,     0.99929,    0.059005,       -2371],
            #                      [  0.042613,    0.050914,     -1.0003,     1.8164],
            #                      [  0,           0,           0,       1]])
            corner_ts = np.dot(transfMtx, np.append(corner_ld, 1))
            corner_ts = [corner_ts[0], -corner_ts[1], -corner_ts[2]]
            print(f'角点NEZ坐标：{corner_ts}')
            logger.info(f'角点NEZ坐标：{corner_ts}')
            # -----------------------------------位姿计算---------------------------------------
            # 测试函数

            # normal_vector = ensure_direction(normal_vector_ts[0:3])
            normal_vector = []
            if np.dot(normal_vector_ts[0:3], [0, -1, 0]) < 0:
                normal_vector = -normal_vector_ts[:3]
            else:
                normal_vector = normal_vector_ts[:3]
            # print(normal_vector)
            # row_vector = np.array(row_vector)
            # 坐标变换后的
            # normal_vector = normal_vector_ts[:3]
            # row_vector = row_vector_ts[:3]

            yaw = yaw_pitch_row(normal_vector, projection_plane='yaw')
            print("偏航角(yaw)：", yaw)
            logger.info(f"偏航角(yaw)：{yaw}")
            pitch = yaw_pitch_row(normal_vector, projection_plane='pitch')
            print("俯仰角(pitch)：", pitch)
            logger.info(f"俯仰角(pitch)：{pitch}")

            print('点云角点x', left_top_point[0][0])
            print('nez坐标e', corner_ts[1])
            # data.append({'点云文件名': pcd_file_path, '图像文件名': image_file_path,
            #              '点云角点x': left_top_point[0], '点云角点y': left_top_point[1],
            #              '点云角点z': left_top_point[2], 'nez坐标n': corner_ts[0],
            #              'nez坐标e': corner_ts[1], 'nez坐标z': corner_ts[3]}, ignore_index=True)

            mqttData = {
                '偏航角': yaw,
                '俯仰角': pitch,
                '点云角点': left_top_point[0].tolist()
            }
            print(mqttData)

            # 发消息
            mqtt_json_data='{"Sys_state":"2","Gps_state":"4","Pitch":"-1.78","Roll":"-0.76","Yaw":"48.80","Ve":"-0.000","Vn":"0.000","Vt":"-0.000","Lon":"117.85765654","Lat":"38.30651157","Hei":"9.78","Pn":"376.226","Pe":"-210.388","Pu":"9.826","Px":"-2341718.598","Py":"4430651.674","Pz":"3932204.047","DeviceId":"32","Qoe":"92"}'
            mqtt_data = json.loads(mqtt_json_data)
            # print(f'mqtt的json发送前数据{mqtt_data},类型为{type(mqtt_data)}')
            mqtt_data["Pitch"] = pitch
            mqtt_data["Roll"] = 0
            mqtt_data["Yaw"] = yaw
            mqtt_data["Pn"] = corner_ts[0]
            mqtt_data["Pe"] = -corner_ts[1]
            mqtt_data["Pu"] = -corner_ts[2]
            # print(f'mqtt的json要发送数据{mqtt_data}')
            sendMessage(mqtt_data)


            # if publish.received_message is not None:
            data = subscriber.received_message

            print(f"多线程接收到的信息: {data}")
                # 处理完消息后，可以将其重置
                # publish.received_message = None




    except Exception as e:
        mqtt_json_data = '{"Sys_state":"2","Gps_state":"4","Pitch":"-1.78","Roll":"-0.76","Yaw":"48.80","Ve":"-0.000","Vn":"0.000","Vt":"-0.000","Lon":"117.85765654","Lat":"38.30651157","Hei":"9.78","Pn":"376.226","Pe":"-210.388","Pu":"9.826","Px":"-2341718.598","Py":"4430651.674","Pz":"3932204.047","DeviceId":"32","Qoe":"92"}'
        mqtt_data = json.loads(mqtt_json_data)
        sendMessage(mqtt_data)

        logger.exception(e)
        pass

def mqtt_thread_function():
    global mqtt_road
    mqtt_thread = subscriber.MqttRoad('127.0.0.1', 1883, 600)
    mqtt_road = mqtt_thread
    mqtt_thread.start()
#
def sendMessage(message):
    if mqtt_road:
        message = json.dumps(message)
        mqtt_road.publish_message('mqtt11', message)
    else:
        print("MQTT 客户端未初始化。")


if __name__ == '__main__':
    # 代码指定了存储点云文件（.pcd）和图像文件（如.jpg或.png）的文件夹路径。
    pcd_folder = r"C:\Users\Yang Yuhao\Desktop\岸基雷视融合代码\船型统计\ld"
    image_folder = r"C:\Users\Yang Yuhao\Desktop\岸基雷视融合代码\船型统计\cam"
    # pcd_folder = r"F:\rslidar"
    # image_folder = r"F:\image"
    # 获取文件夹内所有文件名
    pcd_file_paths = [entry.name for entry in os.scandir(pcd_folder) if entry.is_file()]
    image_file_paths = [entry.name for entry in os.scandir(image_folder) if entry.is_file()]

    # 为mqtt创建一个线程
    mqtt_thread = threading.Thread(target=mqtt_thread_function)
    mqtt_thread.start()

    index = 1
    for pcd_file_path, image_file_path in zip(pcd_file_paths, image_file_paths):
        # if index <= 7:
        #     index = index + 1
        #     continue
        print("-----------------------------------------------------------------")

        print("开始：", time.time())
        print("序号：", index)
        logger.info(f"序号：{index}")
        print(f"点云文件: {pcd_file_path}\n图片文件: {image_file_path}")
        logger.info(f"点云文件: {pcd_file_path}")
        logger.info(f"图片文件: {image_file_path}")

        # 读取.pcd文件，并将其加载为点云对象pcd
        # 使用cv2.imread读取图像文件并存储在image变量中
        pcd = o3d.io.read_point_cloud(os.path.join(pcd_folder + '\\' + pcd_file_path))
        image = cv2.imread(os.path.join(image_folder + '\\' + image_file_path))
        # cv2.imwrite("1.jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        # cv2.imshow("1", image)
        cv2.waitKey(0)

        data = pd.DataFrame(
            columns=['点云文件名', '图像文件名', '点云角点x', '点云角点y', '点云角点z', 'nez坐标n', 'nez坐标e',
                     'nez坐标z'])
        call_back(pcd, image, index, pcd_file_path, image_file_path, data,mqtt_thread)
        index = index + 1

        # cont = input('继续吗?(Y/N)')
        # if cont == 'Y' or cont == 'y':
        #     continue
        # else:
        #     break
