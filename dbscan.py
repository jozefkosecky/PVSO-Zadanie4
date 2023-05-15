import numpy as np
import open3d as o3d
import cv2
import matplotlib.pyplot as plt



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


############### DBSCAN ########################

segment_models = {}
segments = {}

max_plane_idx = 30

rest = pcd
for i in range(max_plane_idx):
    colors = plt.get_cmap("tab20")(i)
    segment_models[i], inliers = rest.segment_plane(
        distance_threshold=0.01, ransac_n=3, num_iterations=1000)
    segments[i] = rest.select_by_index(inliers)

    ############### DBSCAN
    labels = np.array(segments[i].cluster_dbscan(eps=0.05, min_points=10))
    candidates = [len(np.where(labels == j)[0]) for j in np.unique(labels)]
    best_candidate = int(np.unique(labels)[np.where(candidates == np.max(candidates))[0]])

    rest = rest.select_by_index(inliers, invert=True)+segments[i].select_by_index(list(np.where(labels!=best_candidate)[0]))
    segments[i]=segments[i].select_by_index(list(np.where(labels==best_candidate)[0]))
    segments[i].paint_uniform_color(list(colors[:3]))

    print("pass", i, "/", max_plane_idx, "done.")


# ############### DBSCAN one more time
# labels = np.array(rest.cluster_dbscan(eps=0.05, min_points=5))
# max_label = labels.max()
# print(f"point cloud has {max_label + 1} clusters")
# colors = plt.get_cmap("tab10")(labels / (max_label if max_label > 0 else 1))
# colors[labels < 0] = 0
# rest.colors = o3d.utility.Vector3dVector(colors[:, :3])


# o3d.visualization.draw_geometries([segments.values()])
# cv2.waitKey(0)
# o3d.visualization.draw_geometries([segments[i] for i in range(max_plane_idx)]+[rest])
o3d.visualization.draw_geometries([segments[i] for i in range(max_plane_idx)])
# o3d.visualization.draw_geometries([segments[i] for i in range(max_plane_idx)],zoom=0.3199,front=[0.30159062875123849, 0.94077325609922868, 0.15488309545553303],lookat=[-3.9559999108314514, -0.055000066757202148, -0.27599999308586121],up=[-0.044411423633999815, -0.138726419067636, 0.98753122516983349])
# o3d.visualization.draw_geometries([rest])
cv2.waitKey(0)