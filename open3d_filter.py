
import open3d as o3d

import cv2


# https://www.datatechnotes.com/2019/09/clustering-example-with-birch-method-in.html


################ ORIGINAL ###################
# whether to write in binary or text format
write_text = True

# load point cloud from file
file_path = 'TLS_kitchen.ply'

# read point cloud from file
pcd = o3d.io.read_point_cloud(file_path)

pcd.paint_uniform_color([0.6, 0.6, 0.6])

# visualize
o3d.visualization.draw_geometries([pcd])
cv2.waitKey(0)


# http://www.open3d.org/docs/latest/tutorial/Advanced/pointcloud_outlier_removal.html
################# FILTER ###################

print("Load a ply point cloud, print it, and render it")
pcd = o3d.io.read_point_cloud(file_path)
pcd.paint_uniform_color([0.6, 0.6, 0.6])

print("Downsample the point cloud with a voxel of 0.02")
cloud = pcd.voxel_down_sample(voxel_size=0.02)


print("Statistical oulier removal")
cl, ind = cloud.remove_statistical_outlier(nb_neighbors=20,
                                                    std_ratio=1)

inlier_cloud = cloud.select_by_index(ind)
outlier_cloud = cloud.select_by_index(ind, invert=True)

print("Showing outliers (red) and inliers (gray): ")
outlier_cloud.paint_uniform_color([1, 0, 0])
inlier_cloud.paint_uniform_color([0.8, 0.8, 0.8])
o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud],
                                  zoom=0.3412,
                                  front=[0.4257, -0.2125, -0.8795],
                                  lookat=[2.6172, 2.0475, 1.532],
                                  up=[-0.0694, -0.9768, 0.2024])

# write point cloud to file
o3d.io.write_point_cloud("filtered.ply", inlier_cloud, write_ascii=True)
cv2.waitKey(0)
