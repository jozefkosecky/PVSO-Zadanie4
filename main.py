import numpy as np
import open3d as o3d
import os
from matplotlib import pyplot as plt
import threading

# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


# def print_hi(name):
#     # Use a breakpoint in the code line below to debug your script.
#     print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.
#
#
# def use_o3d_test(pts, write_text):
#     pcd = o3d.geometry.PointCloud()
#
#     # the method Vector3dVector() will convert numpy array of shape (n, 3) to Open3D format.
#     # see http://www.open3d.org/docs/release/python_api/open3d.utility.Vector3dVector.html#open3d.utility.Vector3dVector
#     pcd.points = o3d.utility.Vector3dVector(pts)
#
#     # http://www.open3d.org/docs/release/python_api/open3d.io.write_point_cloud.html#open3d.io.write_point_cloud
#     o3d.io.write_point_cloud("my_pts.ply", pcd, write_ascii=write_text)
#
#     # read ply file
#     pcd = o3d.io.read_point_cloud('my_pts.ply')
#
#     # visualize
#     o3d.visualization.draw_geometries([pcd])


def show_original(file_path, write_text):
    # read point cloud from file
    pcd = o3d.io.read_point_cloud(file_path, format='pcd')

    # visualize
    o3d.visualization.draw_geometries([pcd])


# https://www.datatechnotes.com/2019/09/clustering-example-with-birch-method-in.html
def birch():
    pcd = o3d.io.read_point_cloud("filtered.ply")

    labels = np.array(pcd.cluster_dbscan(eps=0.05, min_points=10))

    max_label = labels.max()
    colors = plt.get_cmap("tab20")(labels / (max_label
                                             if max_label > 0 else 1))
    colors[labels < 0] = 0
    pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])
    o3d.visualization.draw_geometries([pcd])

    bclust = Birch(branching_factor=200, threshold=1).fit(x)
    print(bclust)

    labels = bclust.predict(x)

    plt.scatter(x[:, 0], x[:, 1], c=labels)
    plt.show()





def outlier():
    print("Load a ply point cloud, print it, and render it")
    pcd = o3d.io.read_point_cloud("output_big2.pcd")
    # o3d.visualization.draw_geometries([pcd],
    #                                   zoom=0.3412,
    #                                   front=[0.4257, -0.2125, -0.8795],
    #                                   lookat=[2.6172, 2.0475, 1.532],
    #                                   up=[-0.0694, -0.9768, 0.2024])

    print("Downsample the point cloud with a voxel of 0.02")
    voxel_down_pcd = pcd.voxel_down_sample(voxel_size=0.02)
    # o3d.visualization.draw_geometries([voxel_down_pcd],
    #                                   zoom=0.3412,
    #                                   front=[0.4257, -0.2125, -0.8795],
    #                                   lookat=[2.6172, 2.0475, 1.532],
    #                                   up=[-0.0694, -0.9768, 0.2024])

    print("Statistical oulier removal")
    cl, ind = voxel_down_pcd.remove_statistical_outlier(nb_neighbors=20,
                                                        std_ratio=2.0)
    display_inlier_outlier(voxel_down_pcd, ind)


def display_inlier_outlier(cloud, ind):
    inlier_cloud = cloud.select_by_index(ind)
    outlier_cloud = cloud.select_by_index(ind, invert=True)

    print("Showing outliers (red) and inliers (gray): ")
    # outlier_cloud.paint_uniform_color([1, 0, 0])
    # inlier_cloud.paint_uniform_color([0.8, 0.8, 0.8])
    o3d.visualization.draw_geometries([inlier_cloud],
                                      zoom=0.3412,
                                      front=[0.4257, -0.2125, -0.8795],
                                      lookat=[2.6172, 2.0475, 1.532],
                                      up=[-0.0694, -0.9768, 0.2024])

    # write point cloud to file
    o3d.io.write_point_cloud("filtered.ply", inlier_cloud, write_ascii=True)



if __name__ == '__main__':
    # print current working directory
    print(os.getcwd())

    # print contents of directory
    print(os.listdir())

    pts = np.random.randint(0, 100, (100, 3))

    # whether to write in binary or text format
    write_text = True

    # use open3d
    # use_o3d_test(pts, write_text)

    # load point cloud from file
    file_path = 'output_big2.pcd'

    # create threads
    t1 = threading.Thread(target=show_original, args=(file_path, write_text))
    t2 = threading.Thread(target=outlier)
    t3 = threading.Thread(target=birch)

    # start threads
    t3.start()
    t2.start()
    t1.start()

    # wait for threads to finish
    t1.join()
    t2.join()
    t3.join()


