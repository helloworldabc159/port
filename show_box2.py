# import os
# import open3d as o3d
#
# """
# 获取当前工作目录中的所有文件夹
# """
# # 获取当前工作目录
# # current_directory = os.getcwd()
# # 设置目的文件的最外层文件名++
# # current_directory = r"C:\Users\Lenovo\PycharmProjects\pythonProject\cloud_show_continue\show_true2"
# current_directory = r"/home/g/yolov5-master/show"
#
# # 获取当前目录中的所有文件和文件夹
# all_items = os.listdir(current_directory)
#
# # 筛选出所有的文件夹,folders里面包含的是总文件夹下的所有（文件夹的名字）
#
# folders = [item for item in all_items if os.path.isdir(os.path.join(current_directory, item))]
# file_num = len(folders)
#
# # 打印文件夹列表和文件夹数量
# # print("文件夹列表：", folders)
# # print("文件夹数量：", len(folders))
#
# # 拼接文件路径
# # os.path.join(point_cloud_folder, point_cloud_files[i])
#
# for path_pin in folders:
#
#     path = os.path.join(current_directory, path_pin)  # path 存的是小文件夹的（文件路径）
#     print(path)
#     # 获取文件夹中的所有点云文件； 寻找所有以 .ply 结尾的文件名，并将这些文件名存储在一个列表中
#     # point_cloud_files 是一个（列表），里面存了每个小文件夹的（点云文件的文件名）
#     point_cloud_files = [f for f in os.listdir(path) if f.endswith('.ply')]
#
#     # print(point_cloud_files)
#
#     # global_show_arg = 0
#     # point_cloud_files = [f for f in os.listdir(path) if f.endswith('.csv')]
#     # 背景 ＋ 一个人：point_cloud_files = ['colored_all_pcd.ply', 'human_cluster_0.ply']
#     # 背景 ＋ 两个人：point_cloud_files = ['colored_all_pcd.ply', 'human_cluster_0.ply'，'human_cluster_1.ply' ]
#     for name in point_cloud_files:
#
#         # print("\n" + name + "\n")
#
#         if name == 'human_cluster_0.ply':
#
#             # 放一个空的点云容器
#             merged_point_cloud = o3d.geometry.PointCloud()
#
#             show_arg = 1
#
#             # 把背景读进来
#             file_path1 = file_path1 = os.path.join(path, 'colored_all_pcd.ply')
#             pcd1 = o3d.io.read_point_cloud(file_path1)
#
#             # 把第一个人读进来
#             file_path2 = os.path.join(path, 'human_cluster_0.ply')
#             pcd2 = o3d.io.read_point_cloud(file_path2)
#
#             # 给第一个人打框
#
#             axis_aligned_bbox = pcd2.get_axis_aligned_bounding_box()
#             bounding_box = o3d.geometry.LineSet.create_from_axis_aligned_bounding_box(axis_aligned_bbox)
#             bounding_box.paint_uniform_color([1, 0, 0])
#
#             # 合并
#             merged_point_cloud += pcd1
#             merged_point_cloud += pcd2
#
#         if name == 'human_cluster_1.ply':
#             show_arg = 2
#
#             merged_point_cloud2 = o3d.geometry.PointCloud()
#             # file_path1 = file_path1 = os.path.join(path, point_cloud_files[i])
#             # pcd1 = o3d.io.read_point_cloud(file_path1)
#
#             # 把背景读进来
#             file_path3 = file_path3 = os.path.join(path, 'colored_all_pcd.ply')
#             pcd3 = o3d.io.read_point_cloud(file_path3)
#
#             # 把第一个人读进来
#             file_path4 = os.path.join(path, 'human_cluster_0.ply')
#             pcd4 = o3d.io.read_point_cloud(file_path4)
#
#             # 给第一个人打框
#             axis_aligned_bbox = pcd4.get_axis_aligned_bounding_box()
#             bounding_box = o3d.geometry.LineSet.create_from_axis_aligned_bounding_box(axis_aligned_bbox)
#             bounding_box.paint_uniform_color([1, 0, 0])
#
#             merged_point_cloud2 += pcd3
#             merged_point_cloud2 += pcd4
#
#             # 把第二个人读进来
#             file_path4 = os.path.join(path, 'human_cluster_1.ply')
#             pcd5 = o3d.io.read_point_cloud(file_path4)
#
#             axis_aligned_bbox = pcd5.get_axis_aligned_bounding_box()
#             bounding_box2 = o3d.geometry.LineSet.create_from_axis_aligned_bounding_box(axis_aligned_bbox)
#             bounding_box2.paint_uniform_color([1, 0, 0])
#
#
#             merged_point_cloud2 += pcd5
#
#     if show_arg == 1:
#         # print(global_show_arg)
#         o3d.visualization.draw_geometries([merged_point_cloud] + [bounding_box], width=800, height=600)
#     elif show_arg == 2:
#         # print(global_show_arg)
#         o3d.visualization.draw_geometries([merged_point_cloud2] + [bounding_box] + [bounding_box2], width=800,
#                                           height=600)
#
#     # if show_arg == 1:
#     #     # print(global_show_arg)
#     #     o3d.visualization.draw_geometries([merged_point_cloud] , width=800, height=600)
#     # elif show_arg == 2:
#     #     # print(global_show_arg)
#     #     o3d.visualization.draw_geometries([merged_point_cloud2] , width=800,
#     #                                       height=600)
#
#
# """
#     # 逐一读取每两个点云文件，拼接并显示
#     for i in range(0, len(point_cloud_files), 2):
#         # 创建一个空的点云对象
#         merged_point_cloud = o3d.geometry.PointCloud()
#
#         # 读取第一个点云文件
#         file_path1 = os.path.join(path, point_cloud_files[i])
#         pcd1 = o3d.io.read_point_cloud(file_path1)
#
#         # 读取第二个点云文件
#         file_path2 = os.path.join(path, point_cloud_files[i + 1])
#         pcd2 = o3d.io.read_point_cloud(file_path2)
#
#         # 拼接两个点云
#         merged_point_cloud += pcd1
#         merged_point_cloud += pcd2
#
#         # 可以在此处添加其他点云处理操作，如滤波、配准等
#
#         # 显示拼接后的点云
#         o3d.visualization.draw_geometries([merged_point_cloud], width=800, height=600)
#
# """
#
# # axis_aligned_bbox = human_pointcloud.get_axis_aligned_bounding_box()
# #
# # bounding_box = o3d.geometry.LineSet.create_from_axis_aligned_bounding_box(axis_aligned_bbox)
# #
# # bounding_box.paint_uniform_color([1, 0, 0])
# #
# # o3d.visualization.draw_geometries([colored_point_cloud] + [axis_pcd] + [bounding_box])
# # o3d.visualization.draw_geometries([colored_point_cloud] + [axis_pcd] + [bounding_box], window_name='fusion',
# # width=1080, height=800, left=50, top=50, point_show_normal=False, mesh_show_wireframe=False,
# # mesh_show_back_face=False)


import open3d as o3d

file = ""

pcd = o3d.