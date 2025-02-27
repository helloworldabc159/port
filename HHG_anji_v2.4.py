# 添加了log用以过程记录，同时设置一些try结构来防止程序停止运行
import time
import rospy
from sensor_msgs.msg import Image
import message_filters
from sensor_msgs.msg import PointCloud2
from sensor_msgs import point_cloud2
import numpy as np  # 导入模块 numpy，并简写成 np
import sys
import cv2
import os
import open3d as o3d
from pathlib import Path
import torch
import scipy.io
import logging
import subprocess
from datetime import datetime
from logging.handlers import TimedRotatingFileHandler
# ----------------------------------------全局参数-----------------------------------------------------
# 联合标定参数
# 相机内参
K = np.array([[1900.2, 0, 688.723],
              [0, 1911.36, 496.253],
              [0, 0, 1]])
# 旋转矩阵
R = np.array([[0.973976, -0.226649, 0.000702814],
              [0.00405423, 0.0143216, -0.999889],
              [0.226614, 0.973871, 0.0148678]])
# 平移矩阵
t = np.array([-0.137048, -0.00500911, 0.0110638])

# 点云坐标系对象
axis_pcd = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1, origin=[0, 0, 0])
# 从 .mat 文件中加载数据
mat_4x4 = scipy.io.loadmat(r'transfMtx_HHG.mat')
mat_3x3 = scipy.io.loadmat(r'RotfMtx.mat')

script_path = "myscript.sh"

# # 配置日志记录器
# logger = logging.getLogger('test')
# logger.setLevel(logging.INFO)
#
# # 创建文件处理器，使用追加模式
# filename = os.path.join(r"C:\Users\admin\Desktop\yolov5-master\1")
# file_handler = TimedRotatingFileHandler(filename=filename + '\\HHG.log', when="MIDNIGHT", interval=1, backupCount=365)
# file_handler.setLevel(logging.INFO)
#
# # 创建日志格式
# formatter = logging.Formatter('%(asctime)s.%(msecs)03d-%(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
# file_handler.setFormatter(formatter)
#
# # 将处理器添加到记录器
# logger.addHandler(file_handler)
#
# # 定义一个函数，用于创建文件处理器
# def create_file_handler(log_directory):
#     current_time = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
#     log_filename = os.path.join(log_directory, f"{current_time}.log")
#     file_handler = logging.FileHandler(log_filename, mode='a')
#     file_handler.setLevel(logging.INFO)
#     formatter = logging.Formatter('%(asctime)s.%(msecs)03d- %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
#     file_handler.setFormatter(formatter)
#     return file_handler
#
#
# # 指定日志文件保存的目录
# log_directory = 'D:/python/雷视融合日志文件/'
# if not os.path.exists(log_directory):
#     os.makedirs(log_directory)
#
# # 配置日志记录器
# logger = logging.getLogger('test')
# logger.setLevel(logging.INFO)
# # 初始化文件处理器
# file_handler = create_file_handler(log_directory)
# logger.addHandler(file_handler)



# YOLO检测相关内容
# 初始化目录
FILE = Path(__file__).resolve()

ROOT = FILE.parents[0]  # 定义YOLOv5的根目录
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
device = '0'  # 设置代码执行的设备 cuda device, i.e. 0 or 0,1,2,3 or cpu
classes = None  # 在nms中是否是只保留某些特定的类 默认是None 就是所有类只要满足条件都可以保留 --class 0, or --class 0 2 3
agnostic_nms = False  # 进行nms是否也除去不同类别之间的框 默认False
augment = False  # 预测是否也要采用数据增强 TTA 默认False
visualize = False  # 特征图可视化 默认FALSE
half = False  # 是否使用半精度 Float16 推理 可以缩短推理时间 但是默认是False
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
eps = 0.25  # 邻域搜索半径
min_points = 20  # 最小邻域内点的数量

distance_threshold = 0.05  # 阈值用于确定点是否位于平面上
ransac_n = 3  # 用于拟合平面的随机样本点的数量
num_iterations = 1000  # RANSAC算法迭代次数


def run_shell_script(script_path):
    try:
        # 使用subprocess.run来执行shell脚本
        result = subprocess.run(['bash', script_path], check=True, text=True, stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE)
        print("Output:", result.stdout)
        print("Errors:", result.stderr)
    except subprocess.CalledProcessError as e:
        print(f"An error occurred while running the script: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


def Clustering(point_cloud, eps, min_points):
    # print("点云数量：", np.asarray(point_cloud.points).shape)
    # 进行聚类
    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
        labels = np.array(point_cloud.cluster_dbscan(eps=eps, min_points=min_points, print_progress=True))
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


def ensure_direction(vector, reference=np.array([0, -1, 0])):
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


def call_back(ros_pcd, ros_img):
    # print(time.time())
    print("-------------------------------------------------------------------------------------------")
    logger.info("------------------------------------------------------------------")
    try:
        # ----------------------------------------原始数据获取---------------------------------------------------
        # 数据格式转换（point_cloud2转 list 转 np.array）
        print(time.time())
        # 从ROS的ros_pcd消息中读取点云数据，返回一个生成器对象。
        lidar = point_cloud2.read_points(ros_pcd)
        # 将点云数据转换为NumPy数组。
        points = np.array(list(lidar))
        # 仅保留点云中的前三个坐标（x, y, z）。
        points = np.array(points[:, 0:3])
        if points.shape[0] == 0:
            logger.error("激光雷达离线，采集不到点云数据")
            print("激光雷达离线，采集不到点云数据")
            return

        # 记录原始点云数据
        origin_pcd = o3d.geometry.PointCloud()
        origin_pcd.points = o3d.utility.Vector3dVector(points)

        # 过滤点云，得到固定空间内的点云
        # 创建一个新的点云对象，并使用逻辑与操作来过滤点云数据，
        # 只保留特定空间范围内的点。这里的注释说明了对x, y, z坐标的过滤条件。
        point_cloud = o3d.geometry.PointCloud()
        after_filter = np.logical_and.reduce(
            [
                points[:, 0] >= 0, points[:, 0] <= 40,  # 水平方向
                # points[:, 1] >= 5,    # 纵深
                # points[:, 2] >= -10.5, points[:, 2] <= 1, # 垂直方向
            ]
        )
        point_cloud.points = o3d.utility.Vector3dVector(
            points[after_filter]
        )
        print("检测范围内点云数量：", np.asarray(point_cloud.points).shape)

        # 记录原始图像数据
        # if np.asarray(ros_img.data).shape[0] == 0:
        #     logger.error("相机离线，采集不到图像数据")
        #     print("相机离线，采集不到图像数据")
        #     return

        # 将ROS图像消息转换为NumPy数组，并重新排列数组以匹配图像的宽度和高度。
        # 然后使用OpenCV库将RGB格式的图像转换为BGR格式
        image = np.frombuffer(ros_img.data, dtype=np.uint8).reshape(720, 1280, -1)  # rgb格式
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # ---------------------------------------yolo识别定位-------------------------------------------------

        # 调用detect函数，传入图像数据，返回检测到的物体列表。这个函数可能是使用YOLO（You Only Look
        # Once）或其他目标检测算法来识别图像中的物体。
        detections = detect(image)

        # 如果检测到的物体数量为0，记录一个警告日志说明YOLO未识别到船只，然后返回（函数结束）。
        # 这里有一个被注释掉的调用shell脚本的代码，如果需要执行其他操作，可以取消注释。
        if len(detections) == 0:
            logger.warning("YOLO未识别到船只")
            # run_shell_script(script_path)
            return

        # 打印和记录检测到的物体列表。f"识别结果：{detections}"
        # 使用了Python的f - string格式化字符串，将变量detections的内容插入到字符串中
        print("识别结果：", detections)
        logger.info(f"识别结果：{detections}")
        i = 0
        objects = []
        for detection in detections:
            # 在这个循环内部，首先创建了一个名为obj的新字典，用于存储当前检测到的物体的信息。这个字典有一个键
            # 'class'，其值是当前检测结果的'class'键对应的值，这通常是物体的类别名称（例如，"boat"表示船只）。
            # ？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？
            obj = {'class': detection['class']}
            real_coordinate = (detection['position'][0], detection['position'][0] + detection['position'][2],
                               detection['position'][1], detection['position'][1] + detection['position'][3])

            # 将计算出的实际坐标元组real_coordinate作为值赋给obj字典的键
            # 'real_coordinate'。
            obj['real_coordinate'] = real_coordinate
            # ？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？
            i = i + 1

            # 最后，将obj字典添加到objects列表中。这个列表会随着循环的进行而增长，最终包含所有检测到的物体的信息。
            objects.append(obj)
        # print(objects)

        # ----------------------------------------数据融合-----------------------------------------------------
        # 图像与雷达 数据层融合
        cloud = np.asarray(point_cloud.points)
        colored_points = []
        for point in cloud:
            # 点云转换到相机坐标系
            # 这里创建了一个NumPy数组pL，它代表三维空间中的一个点point，并且增加了一个维度，用于齐次坐标表示。齐次坐标是一种在数学中常用的方式，用于表示射影空间中的点，
            # 其中第四个坐标通常是1。这里point是一个包含三个元素的列表或元组，分别代表点的x, y, z坐标。
            pL = np.array([point[0], point[1], point[2], 1.0])
            # 这一行代码执行了两个操作：首先，使用 @ 操作符对R（一个3x3旋转矩阵）和pL的前三个元素（即三维空间坐标）进行矩阵乘法，这表示将点pL绕某个轴旋转；
            # 然后，将旋转后的结果加上平移向量t（一个三维向量），从而得到点pL在相机坐标系中的位置pC。
            pC = R @ pL[:3] + t

            # 投影到图像平面
            # 这里，K是一个3x3的相机内参矩阵，它包含了相机的焦距和主点坐标等信息。通过对相机坐标系中的点pC和内参矩阵K进行矩阵乘法，我们得到了点pC在归一化图像平面上的坐标q。
            # 归一化图像平面是一个虚拟的平面，其中x和y坐标与图像的像素坐标成比例，而z坐标代表深度信息。
            q = K @ pC
            # 最后，这两行代码将归一化图像平面上的点q转换为实际的图像像素坐标。由于q是一个齐次坐标，我们需要通过除以q[2]（即z坐标）来消除齐次坐标的第四个维度，从而得到二维图像平面上的坐标。
            # 这里的u和v分别代表图像的横坐标（列）和纵坐标（行）。通过取整（int()），我们将坐标转换为整数，因为在图像中像素坐标必须是整数。这样，我们就得到了点point在图像上的投影位置
            # u = int(q[0] / q[2])
            v = int(q[1] / q[2])

            # 检查是否在图像范围内
            if (obj['real_coordinate'][0] < u < obj['real_coordinate'][1] - 5) and \
                    (obj['real_coordinate'][2] < v < obj['real_coordinate'][3] - 5):
                color = image[v, u]  # 提取图像中的颜色信息
                # 将带有颜色信息的点添加到列表中。颜色信息是归一化的，即除以255。
                colored_points.append(
                    [point[0], point[1], point[2], color[2] / 255.0, color[1] / 255.0, color[0] / 255.0])  # RGB归一化
        # 创建一个Open3D点云对象用于存储带颜色的点云。
        colored_cloud = o3d.geometry.PointCloud()
        if len(colored_points) == 0:  # 创建一个Open3D点云对象用于存储带颜色的点云。
            logger.error("YOLO wu dian yun")
            return
        # 将点和颜色信息设置到Open3D点云对象中。
        colored_cloud.points = o3d.utility.Vector3dVector(np.array(colored_points)[:, :3])
        colored_cloud.colors = o3d.utility.Vector3dVector(np.array(colored_points)[:, 3:])
        # o3d.visualization.draw_geometries([colored_cloud, point_cloud], window_name="融合后彩色点云")

        # ---------------------------------------船尾面拟合-----------------------------------------------------
        # print(time.time())
        # 这行代码使用Open3D库中的segment_plane函数来从colored_cloud点云中分割出一个平面。distance_threshold是点到平面的最大距离阈值，用于判断一个点是否属于平面。ransac_n是随机采样一致性算法（RANSAC）中每次迭代所用的点的数量。num_iterations是RANSAC算法迭代的次数。
        # 函数返回平面模型参数plane_model和一个布尔数组inliers，表示哪些点属于平面。
        plane_model, inliers = colored_cloud.segment_plane(distance_threshold=0.005, ransac_n=3, num_iterations=10000)
        # print(time.time())

        # 这里，平面模型参数plane_model是一个四元组，代表平面方程ax + by + cz + d = 0
        # 中的系数a、b、c和d。代码将这个四元组解包到变量a、b、c和d中，并打印出平面方程。
        [a, b, c, d] = plane_model
        print("YOLO识别面方程参数:")
        print(f"Plane equation: {a:.6f}x + {b:.6f}y + {c:.6f}z + {d:.6f} = 0")
        # ----------------------按YOLO识别结果过滤原始点云-----------------------------------------------------

        # 这两行代码将原始点云point_cloud中的点转换为NumPy数组raw_points，并创建一个新的Open3D点云对象tail_point_cloud。
        raw_points = np.asarray(point_cloud.points)
        tail_point_cloud = o3d.geometry.PointCloud()

        # 这段代码创建了一个布尔数组after_filter，它表示原始点云中的点是否满足两个条件：点到分割出的平面的距离在 - 0.3
        # 到0.3之间。这是通过逻辑与操作np.logical_and.reduce来实现的。
        after_filter = np.logical_and.reduce([
            # params['a'] * raw_points[:, 0] + params['b'] * raw_points[:, 1] + params['c'] * raw_points[:, 2] + params[
            #     'd'] >= -0.3,
            # params['a'] * raw_points[:, 0] + params['b'] * raw_points[:, 1] + params['c'] * raw_points[:, 2] + params[
            #     'd'] <= 0.3
            a * raw_points[:, 0] + b * raw_points[:, 1] + c * raw_points[:, 2] + d >= -0.3,
            a * raw_points[:, 0] + b * raw_points[:, 1] + c * raw_points[:, 2] + d <= 0.3
        ])
        # 这两行代码将满足过滤条件的点添加到tail_point_cloud点云中，并用红色（RGB值为[1, 0, 0]）统一上色。
        tail_point_cloud.points = o3d.utility.Vector3dVector(raw_points[after_filter])
        tail_point_cloud.paint_uniform_color([1, 0, 0])
        # o3d.visualization.draw_geometries([tail_point_cloud], window_name="船尾平面点云")
        # o3d.io.write_point_cloud("tail.pcd", tail_point_cloud)
        # ----------------------按YOLO识别结果过滤原始点云-----------------------------------------------------
        # print(time.time())

        # 最后这部分代码对过滤后的点云tail_point_cloud再次执行平面分割操作，
        # 并打印出新的平面方程参数。这些参数同样被记录到日志中。
        # 使用RANSAC算法从tail_point_cloud点云中分割出一个平面。
        # 在每次迭代中，随机选择3个点来估计平面模型。
        # 最多进行100000次迭代来找到最佳的平面模型。
        # 将那些与估计平面距离小于或等于0
        # .005
        # 的点视为平面的一部分。
        # plane_model将包含所估计平面的参数（通常是平面方程的系数）。
        # inliers将是一个布尔数组，标记了哪些点属于分割出的平面。
        plane_model, inliers = tail_point_cloud.segment_plane(distance_threshold=0.005, ransac_n=3,
                                                              num_iterations=100000)
        # print(time.time())
        [a, b, c, d] = plane_model
        print("船尾面平面方程参数:")
        print(f"Plane equation: {a:.6f}x + {b:.6f}y + {c:.6f}z + {d:.6f} = 0")
        logger.info("船尾面平面方程参数:")
        logger.info(f"Plane equation: {a:.6f}x + {b:.6f}y + {c:.6f}z + {d:.6f} = 0")
        # ----------------------特征点提取-----------------------------------
        # 聚类

        # 用于对点云tail_point_cloud进行聚类操作。eps参数定义了聚类时使用的邻域大小，
        # 、即两个点被认为是同一个簇成员的最大距离。min_points参数定义了形成簇所需的最小点数。如果在执行聚类过程中发生任何异常，它将被捕获。
        #

        try:
            corner_pcd = Clustering(tail_point_cloud, eps=0.3, min_points=20)
        # 这是一个异常处理块，如果Clustering函数在执行过程中抛出任何异常，该异常将被捕获，
        # 异常信息将被记录到日志中（使用logger.warning），然后函数执行返回，不再执行后续代码
        except Exception as e:
            logger.warning(e)
            return
        # o3d.visualization.draw_geometries([corner_pcd, axis_pcd], window_name="聚类1")
        if corner_pcd == "error":
            print("第一次聚类出错")
            logger.warning("第一次聚类出错")
            return

        # 这行代码将聚类后的点云corner_pcd中的点转换为NumPy数组tpcd_points，以便进行后续的数组操作
        tpcd_points = np.asarray(corner_pcd.points)
        # 这两行代码分别获取聚类后点云的最小边界min_val和最大边界max_val，
        # 它们是三维向量，分别代表点云在x、y、z方向上的最小值和最大值
        min_val = corner_pcd.get_min_bound()
        max_val = corner_pcd.get_max_bound()
        # print(min_val[0], max_val[0])

        # 提取ROI
        # 这行代码创建了一个新的Open3D点云对象tail_ROI_pcd，用于存储感兴趣区域（ROI）的点。
        tail_ROI_pcd = o3d.geometry.PointCloud()

        # 这行代码创建了一个布尔数组after_filter，用于过滤点云。它使用NumPy的逻辑与操作np.logical_and.reduce，确保只有同时满足两个条件的点才会被保留：点的x坐标大于等于最小边界min_val[
        #     0]，并且小于等于min_val[0]
        # 加上max_val[0]
        # 和min_val[0]
        # 之间距离的1 / 7。          ？？？？？？？？？？？？？？？？？？？？？
        after_filter = np.logical_and.reduce(
            [
                tpcd_points[:, 0] >= min_val[0], tpcd_points[:, 0] <= min_val[0] + (max_val[0] - min_val[0]) / 7
            ]
        )

        # 最后，这行代码将过滤后的点（满足ROI条件的点）添加到tail_ROI_pcd点云对象中，这样tail_ROI_pcd就只包含ROI内的点了。
        # o3d.utility.Vector3dVector是将NumPy数组转换为Open3D点云可以接受的格式。
        tail_ROI_pcd.points = o3d.utility.Vector3dVector(
            tpcd_points[after_filter]
        )

        # 聚类
        # try:
        #     corner_pcd = Clustering(tail_ROI_pcd, eps, min_points)
        # o3d.visualization.draw_geometries([corner_pcd, axis_pcd], window_name="聚类2")
        # except Exception as e:
        #     logger.warning(e)
        #     print("第二次聚类出错")
        #     return
        # if corner_pcd == "error":
        #     logger.warning("第二次聚类出错")
        #     print("第二次聚类出错")
        #     return
        # 获取点云的点坐标
        corner_pcd = tail_ROI_pcd
        corner_points = np.asarray(corner_pcd.points)
        # --------------------------找到最左上角的角点-----------------------------
        # # 方案一
        # # 最小x坐标
        # min_x_index = np.argmin(corner_points[:, 0])
        # left_point_index = np.where((corner_points[:, 0] == corner_points[min_x_index, 0]))
        # left_point = corner_points[left_point_index]
        # # 最大z坐标
        # max_z_index = np.argmax(left_point[:, 2])
        # top_point_index = np.where((left_point[:, 2] == left_point[max_z_index, 2]))
        # left_top_point = left_point[top_point_index]

        # 方案二
        # 取反x坐标
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
        # o3d.visualization.draw_geometries([centroid_sphere, point_cloud], window_name="角点提取")

        # -----------------------------------坐标变换---------------------------------------
        # 获取变量
        transfMtx = mat_3x3['transfMtx']
        # ld2ts
        normal_vector_ld = np.array([a, b, c])
        # normal_vector_ld = np.array([tail_params['a'], tail_params['b'], tail_params['c']])
        normal_vector_ts = np.dot(transfMtx, np.append(normal_vector_ld, 1))
        corner_ld = np.asarray(corner.points)
        transfMtx = mat_4x4['transfMtx']
        corner_ts = np.dot(transfMtx, np.append(corner_ld, 1))
        corner_ts = [corner_ts[0], -corner_ts[1], -corner_ts[2]]
        print(f'角点NEZ坐标：{corner_ts}')
        logger.info(f'角点NEZ坐标：{corner_ts}')

        # -----------------------------------位姿计算---------------------------------------
        # 测试函数
        if np.dot(normal_vector_ts[0:3], [0, -1, 0]) < 0:
            normal_vector = -normal_vector_ts[:3]
        else:
            normal_vector = normal_vector_ts[:3]
        yaw = yaw_pitch_row(normal_vector, projection_plane='yaw')
        print("偏航角(yaw)：", yaw)
        logger.info(f"偏航角(yaw)：{yaw}")
        pitch = yaw_pitch_row(normal_vector, projection_plane='pitch')
        print("俯仰角(pitch)：", pitch)
        logger.info(f"俯仰角(pitch)：{pitch}")
    except Exception as e:
        logger.exception(e)
        pass


if __name__ == '__main__':
    rospy.init_node('cloud_img_process', anonymous=True)
    print('初始化')

    ros_pcd = message_filters.Subscriber("/lslidar_point_cloud", PointCloud2, queue_size=1)
    ros_img = message_filters.Subscriber("/image1", Image, queue_size=1)
    ts = message_filters.ApproximateTimeSynchronizer([ros_pcd, ros_img], 10, 0.1, allow_headerless=True)
    ts.registerCallback(call_back)

    # 循环来源
    rospy.spin()  # spin() simply keeps python from exiting until this node is stopped
