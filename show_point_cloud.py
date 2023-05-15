import open3d as o3d
import cv2

# https://www.datatechnotes.com/2019/09/clustering-example-with-birch-method-in.html



################ ORIGINAL ###################
# whether to write in binary or text format
write_text = True

# load point cloud from file
file_path = 'output_big3.pcd'

# read point cloud from file
pcd = o3d.io.read_point_cloud(file_path)
# pcd.paint_uniform_color([1, 0, 0])
# visualize
o3d.visualization.draw_geometries([pcd])
cv2.waitKey(0)
