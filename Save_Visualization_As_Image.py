import os

import open3d as o3d


def save_visualization_as_image(geometry, filename,view_params=None):
    """
    将几何体可视化并保存为图片。

    参数:
    geometry: Open3D 几何体对象（如点云、网格等）。
    filename: 保存图片的文件名（包括路径和扩展名）。
    """
    # 初始化可视化器
    vis = o3d.visualization.Visualizer()
    vis.create_window()

    # 添加几何体到可视化器
    vis.add_geometry(geometry)

    # 获取视图控制器
    ctr = vis.get_view_control()

    # 设置视角
    ctr.set_front([1, 1, -1])  # 向左转90度并往下转90度
    ctr.set_lookat([0, 0, 0])  # 观察目标
    ctr.set_up([0, 0, 1])  # 上方方向
    ctr.set_zoom(1)  # 缩放

    # 渲染并保存图片
    vis.poll_events()
    vis.update_renderer()
    vis.capture_screen_image(filename)  # 保存为图片

    # 关闭可视化窗口
    vis.destroy_window()
if __name__ == '__main__':
    path = r"C:\Users\Yang Yuhao\Desktop\找点云工作\分析角点跳动原因\test"
    if not os.path.exists(path):
        print("指定的路径不存在，请检查。")
