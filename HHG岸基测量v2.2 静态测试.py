# 更替了平面拟合的程序，缩短了程序运行时间
import time

import numpy as np  # 导入模块 numpy，并简写成 np

import sys
# sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')     # 终端执行时候需要加入，否则报错，将ros依赖的python2去除
import cv2
import os
import open3d as o3d
from pathlib import Path
import torch
import scipy.io

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
mat_4x4 = scipy.io.loadmat(r'D:\python\python1\yolov5-master\transfMtx_HHG.mat')
mat_3x3= scipy.io.loadmat(r'D:\python\python1\yolov5-master\RotfMtx.mat')
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

def Clustering(point_cloud, eps, min_points):
    # print("点云数量：", np.asarray(point_cloud.points).shape)
    # 进行聚类
    # with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
    #     labels = np.array(point_cloud.cluster_dbscan(eps=eps, min_points=min_points, print_progress=True))
    labels = np.array(point_cloud.cluster_dbscan(eps=eps, min_points=min_points, print_progress=True))
    if len(labels) > 0:
        max_label = max(labels)
        min_label = min(labels)
        # print(f"label:{min_label}~{max_label}")
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
    projection_vector = normal_vector - np.dot(normal_vector, plane_normal) / np.linalg.norm(plane_normal)**2 * plane_normal

    # 计算投影向量与轴的夹角
    cos_angle = np.dot(projection_vector, axis_vector) / (np.linalg.norm(projection_vector) * np.linalg.norm(axis_vector))


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


def call_back(pcd, img):
    print(time.time())
    # print("-------------------------------------------------------------------------------------------")
    # ----------------------------------------原始数据获取---------------------------------------------------
    points = np.asarray(pcd.points)
    # 记录原始点云数据
    # origin_pcd = o3d.geometry.PointCloud()
    # origin_pcd.points = o3d.utility.Vector3dVector(points)
    # 过滤点云，得到固定空间内的点云
    point_cloud = o3d.geometry.PointCloud()
    after_filter = np.logical_and.reduce(
        [
            points[:, 0] >= 0, points[:, 0] <= 20,  # 水平方向
            # points[:, 1] >= 2, points[:, 1] <= 40,    # 纵深
            # points[:, 2] >= -10.5, points[:, 2] <= 1, # 垂直方向
        ]
    )
    point_cloud.points = o3d.utility.Vector3dVector(
        points[after_filter]
    )
    print("过滤后点云数量：", np.asarray(point_cloud.points).shape)
    # 记录原始图像数据
    image = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    # print(image.shape)
    # ---------------------------------------yolo识别定位-------------------------------------------------
    detections = detect(image)
    print("识别结果：", detections)
    i = 0
    objects = []
    for detection in detections:
        obj = {'class': detection['class']}
        real_coordinate = (detection['position'][0], detection['position'][0] + detection['position'][2],
                           detection['position'][1], detection['position'][1] + detection['position'][3])
        obj['real_coordinate'] = real_coordinate
        i = i + 1
        objects.append(obj)
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
            u = int(q[0] / q[2])
            v = int(q[1] / q[2])

            # 检查是否在图像范围内
            if (obj['real_coordinate'][0] <= u <= obj['real_coordinate'][1]) and \
                (obj['real_coordinate'][2] <= v <= obj['real_coordinate'][3]):
                color = image[v, u]  # 提取图像中的颜色信息
                colored_points.append(
                    [point[0], point[1], point[2], color[2] / 255.0, color[1] / 255.0, color[0] / 255.0])  # RGB归一化
        colored_cloud = o3d.geometry.PointCloud()
        colored_cloud.points = o3d.utility.Vector3dVector(np.array(colored_points)[:, :3])
        colored_cloud.colors = o3d.utility.Vector3dVector(np.array(colored_points)[:, 3:])
        # o3d.io.write_point_cloud("yolo_color.pcd", colored_cloud)
        # o3d.visualization.draw_geometries([colored_cloud, point_cloud], window_name="融合后彩色点云")
        # ---------------------------------------拟合-----------------------------------------------------
        if np.asarray(colored_cloud.points).shape[0] <= 3:
            print("错误，tail_pcd内点数小于3")
        try:
            params, inliers = colored_cloud.segment_plane(distance_threshold=0.005, ransac_n=3, num_iterations=10000)
            # 打印平面方程参数
            print("YOLO识别面方程参数:")
            print(f"a: {params[0]}, b: {params[1]}, c: {params[2]}, d: {params[3]}")
        except:
            pass
        # ----------------------按YOLO识别结果过滤原始点云-----------------------------------------------------
        raw_points = np.asarray(point_cloud.points)
        tail_point_cloud = o3d.geometry.PointCloud()
        after_filter = np.logical_and.reduce([
            params[0] * raw_points[:, 0] + params[1] * raw_points[:, 1] + params[2] * raw_points[:, 2] + params[3] >= -0.3,
            params[0] * raw_points[:, 0] + params[1] * raw_points[:, 1] + params[2] * raw_points[:, 2] + params[3] <= 0.3
        ])
        tail_point_cloud.points = o3d.utility.Vector3dVector(raw_points[after_filter])
        tail_point_cloud.paint_uniform_color([1, 0, 0])
        # o3d.visualization.draw_geometries([tail_point_cloud], window_name="船尾平面点云")
        # o3d.io.write_point_cloud("tail.pcd", tail_point_cloud)
        # ----------------------按YOLO识别结果过滤原始点云-----------------------------------------------------

        tail_params, tail_inliers = tail_point_cloud.segment_plane(distance_threshold=0.005, ransac_n=3, num_iterations=10000)
        # 绘制结果
        # 打印平面方程参数
        print("船尾面平面方程参数:")
        print(f"a: {tail_params[0]}, b: {tail_params[1]}, c: {tail_params[2]}, d: {tail_params[3]}")
        # ----------------------特征点提取-----------------------------------
        # 读取点云
        # 聚类
        corner_pcd = Clustering(tail_point_cloud, eps=0.3, min_points=20)

        # o3d.visualization.draw_geometries([corner_pcd, axis_pcd], window_name="聚类1")

        tpcd_points = np.asarray(corner_pcd.points)
        min_val = corner_pcd.get_min_bound()
        max_val = corner_pcd.get_max_bound()

        # 提取ROI
        tail_ROI_pcd = o3d.geometry.PointCloud()
        after_filter = np.logical_and.reduce(
            [
                tpcd_points[:, 0] >= min_val[0], tpcd_points[:, 0] <= min_val[0] + (max_val[0] - min_val[0]) / 7
            ]
        )
        tail_ROI_pcd.points = o3d.utility.Vector3dVector(
            tpcd_points[after_filter]
        )

        # o3d.io.write_point_cloud("zzz.pcd", tail_ROI_pcd)

        # 聚类
        corner_pcd = Clustering(tail_ROI_pcd, eps, min_points)
        # o3d.visualization.draw_geometries([corner_pcd, axis_pcd], window_name="聚类2")
        # 获取点云的点坐标
        corner_points = np.asarray(corner_pcd.points)

        # --------------------------找到最左上角的角点-----------------------------
        # 方案一
        # 最小x坐标
        min_x_index = np.argmin(corner_points[:, 0])
        left_point_index = np.where((corner_points[:, 0] == corner_points[min_x_index, 0]))
        # print(left_point_index)
        left_point = corner_points[left_point_index]
        # print(left_point)
        # 最大z坐标
        max_z_index = np.argmax(left_point[:, 2])
        top_point_index = np.where((left_point[:, 2] == left_point[max_z_index, 2]))
        # print(top_point_index)
        left_top_point = left_point[top_point_index]

        # # 方案二
        # # 取反x坐标
        # neg_x = -corner_points[:, 0]
        # # 计算加权和(加权因子可调)
        # weighted_sum = 0.25 * neg_x + 0.75 * corner_points[:, 2]
        # # 找到和最大的点的索引
        # max_index = np.argmax(weighted_sum)
        # # 提取和最大的点
        # left_top_point = corner_points[max_index]
        # left_top_point = [left_top_point]


        corner = o3d.geometry.PointCloud()
        corner.points = o3d.utility.Vector3dVector(left_top_point)
        corner.paint_uniform_color([1, 0, 0])
        print("角点点云坐标：", np.asarray(corner.points))

        centroid_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.1)
        centroid_sphere.translate(left_top_point[0])
        centroid_sphere.paint_uniform_color([1, 0, 0])  # 质心上色为红色

        # o3d.visualization.draw_geometries([centroid_sphere, point_cloud], window_name="角点提取")


        # -----------------------------------坐标变换---------------------------------------
        # 获取变量
        transfMtx = mat_3x3['transfMtx']
        # ld2ts
        normal_vector_ld = np.array([tail_params[0], tail_params[1], tail_params[2]])
        # print(np.append(normal_vector_ld, 1).shape)
        normal_vector_ts = np.dot(transfMtx, np.append(normal_vector_ld, 1))
        # print("ts", normal_vector_ts)
        # print(normal_vector_ts.shape)
        corner_ld = np.asarray(corner.points)
        transfMtx = mat_4x4['transfMtx']
        corner_ts = np.dot(transfMtx, np.append(corner_ld, 1))
        corner_ts = [corner_ts[0], -corner_ts[1], -corner_ts[2]]
        print(f'角点NEZ坐标：{corner_ts}')

        #-----------------------------------位姿计算---------------------------------------
        # 测试函数
        if np.dot(normal_vector_ts[0:3], [0, -1, 0]) < 0:
            normal_vector = -normal_vector_ts[:3]
        else:
            normal_vector = normal_vector_ts[:3]
        yaw = yaw_pitch_row(normal_vector, projection_plane='yaw')
        print("偏航角(yaw)：", yaw)

        pitch = yaw_pitch_row(normal_vector, projection_plane='pitch')
        print("俯仰角(pitch)：", pitch)
        print(time.time())



# if __name__ == '__main__':
#     pcd = o3d.io.read_point_cloud(r"C:\Users\wxw\Desktop\yolov5-master\pcd data\Tailcloud2024-05-11 12_37_34.pcd")
#     image = cv2.imread(r"C:\Users\wxw\Desktop\yolov5-master\pcd data\2024-05-11_12_37_03.jpg")
#     call_back(pcd, image)
#
# if __name__ == '__main__':
#     pcd = o3d.io.read_point_cloud(r"C:\Users\wxw\Desktop\yolov5-master\pcd data\Tailcloud2024-05-11 12_37_34.pcd")
#     image = cv2.imread(r"C:\Users\wxw\Desktop\yolov5-master\pcd data\2024-05-11_12_37_03.jpg")
#     call_back(pcd, image)

# 0822 第一次测试
# if __name__ == '__main__':
#     pcd = o3d.io.read_point_cloud(r"C:\Users\wxw\Desktop\yolov5-master\originData\20240822113621.pcd")
#     image = cv2.imread(r"C:\Users\wxw\Desktop\yolov5-master\105save\frame_20240822_113621_0016.jpg")
#     call_back(pcd, image)


if __name__ == '__main__':
    pcd = o3d.io.read_point_cloud(r"C:\Users\wxw\Desktop\yolov5-master\originData\20240822113558.pcd")
    image = cv2.imread(r"C:\Users\wxw\Desktop\yolov5-master\105save\frame_20240822_113558_0005.jpg")
    call_back(pcd, image)
