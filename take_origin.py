import time
import open3d as o3d
import rospy
from sensor_msgs.msg import PointCloud2
from sensor_msgs import point_cloud2
import numpy as np

class PointCloudSubscriber(object):
    def __init__(self) -> None:
        self.sub = rospy.Subscriber("/lslidar_point_cloud",
                                     PointCloud2,
                                     self.callback, queue_size=5)
    def callback(self, msg):
        assert isinstance(msg, PointCloud2)
        points = []
        for msg in point_cloud2.read_points_list(msg,
                                    field_names=('x', 'y', 'z', 'intensity'),
                                    skip_nans=True):
            print(msg)
            points.append([msg[0], msg[1], msg[2], msg[3]])
        points = np.array(points)
        print(points)
        # pcd = o3d.geometry.PointCloud()
        # pcd.points = o3d.utility.Vector3dVector(points[:, 0:3])
        # pcd.points['intensity'] = o3d.utility.Vector3dVector(points[:, 3])
        # o3d.io.write_point_cloud('1.pcd', pcd)

        device = o3d.core.Device('CPU:0')
        dtype = o3d.core.float32
        pcd = o3d.t.geometry.PointCloud(device)
        pcd.point.positions = o3d.core.Tensor(points[:, 0:3], dtype, device)
        pcd.point.intensity = o3d.core.Tensor(points[:, 3], dtype, device)

        o3d.t.io.write_point_cloud("1.pcd", pcd, write_ascii=True)
        # gen=point_cloud2.read_points(msg,field_names=("x","y","z"))
        # print(time.time())
        # print(points)
        # np.savetxt('RS.csv', points, delimiter = ",")


def callback(lidar):
    lidar = point_cloud2.read_points(lidar)
    points = np.array(list(lidar))
    print(points[:, 0:3])
    print(time.time())

#
if __name__ =='__main__':
    rospy.init_node("pointcloud_subscriber")
    PointCloudSubscriber()
    rospy.spin()

# if __name__ =='__main__':
#     rospy.init_node("pointcloud_subscriber")
#     rospy.Subscriber("/lslidar_point_cloud", PointCloud2, callback)
#     rospy.spin()