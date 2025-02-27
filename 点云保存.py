import numpy as np
import rospy
from sensor_msgs import point_cloud2
from sensor_msgs.msg import PointCloud2
from sensor_msgs.msg import Image
import open3d as o3d
from cv_bridge import CvBridge
import time
import cv2
def save_point_cloud_st(msg):
    points = []
    # 速腾
    for msg in point_cloud2.read_points_list(msg,
                                             field_names=('x', 'y', 'z','intensity'),
                                             skip_nans=True):
        points.append([msg[0], msg[1], msg[2]])
    points = np.array(points)
    return points

def save_point_cloud_ls(self, msg):
    # 镭神
    lidar = point_cloud2.read_points(msg)
    points = np.array(list(lidar))
    points = np.array(points[:, :3])  # 获取前三列数据

    return points






def save_image(msg):
    bridge = CvBridge()
    # 将 ROS 图像消息转换为 OpenCV 格式
    cv_image = bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

    # 获取当前时间戳
    t = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())

    # 保存图像
    filename = f"images/{t}.png"
    cv2.imwrite(filename, cv_image)
    print(f'已保存图像：{filename}')


if __name__ == "__main__":
    rospy.init_node('image_saver', anonymous=True)

    # 等待一次图像消息
    points_msg = rospy.wait_for_message("/camera/image_raw", Image)
    save_image(points_msg)

if __name__ == "__main__":

    rospy.init_node('cloud', anonymous=True)

    while True:
        points_msg = rospy.wait_for_message("/middle/rslidar_points", PointCloud2)
        # print(points_msg)
        points = save_point_cloud_st(points_msg)


        t = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)


        o3d.io.write_point_cloud(f"pcd1//{t}.pcd", pcd)
        print(f'已保存{t}时刻点云')
