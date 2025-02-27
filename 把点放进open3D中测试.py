import open3d as o3d
import numpy as np
import json

# 假设你的 JSON 数据是这样的
json_data = '''[
    [46.0909462, -1.778691888, 4.076000214],
    [46.09751511, -1.448672771, 4.075560093],
    [46.04916763, -1.278226376, 4.054638386],
    [46.08840179, -1.110278964, 4.057706833],
    [46.04038239, -0.7795240879, 4.052884579],
    [45.93336487, -0.609318912, 4.04324007]
]'''
try:
    with open('newdata.json', 'r') as f:
        data = json.load(f)

    # print('读取的数据:', data)

except FileNotFoundError:
    print('错误: data.json 文件未找到。')
except json.JSONDecodeError:
    print('错误: JSON 解码失败。请检查文件内容。')
except Exception as e:
    print(f'发生错误: {e}')


# 提取每个元素的第一个和第三个值
points = np.array([[item[1], item[2], 0] for item in data])  # 保持三维，z坐标为0

# 创建一个可视化器
vis = o3d.visualization.Visualizer()
vis.create_window()

# 创建一个点云对象
point_cloud = o3d.geometry.PointCloud()

# 将提取的点赋值给点云
point_cloud.points = o3d.utility.Vector3dVector(points)

# 将点云添加到可视化器
vis.add_geometry(point_cloud)

# 开始可视化
vis.run()

# 关闭可视化器
vis.destroy_window()
