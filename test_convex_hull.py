import open3d as o3d

pcd = o3d.io.read_point_cloud(
    'C:\\Users\\Yang Yuhao\\Desktop\\岸基雷视融合代码\\船型统计\\ld\\Tailcloud2023-04-03 13_49_20.pcd')
print(pcd)

hull, idx = pcd.compute_convex_hull()
hull_cloud = pcd.select_by_index(idx)
print(hull, idx)
o3d.io.write_point_cloud('hull_cloud.pcd', hull_cloud)
hull_ls = o3d.geometry.LineSet.create_from_triangle_mesh(hull)
hull_ls.paint_uniform_color((1, 0, 0))
o3d.visualization.draw_geometries([pcd,hull_ls])
