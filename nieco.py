import ctypes
import _ctypes

import pygame
import sys
import numpy as np

# Import PyKinect2 constants
from pykinect2 import PyKinectV2
from pykinect2.PyKinectRuntime import PyKinectRuntime
from pykinect2.PyKinectV2 import *

# Initialize the Kinect sensor
kinect = PyKinectRuntime(PyKinectV2.FrameSourceTypes_Depth)

# Capture depth data
depth_frame = kinect.get_last_depth_frame()

# Convert depth data to a NumPy array
depth_image = depth_frame.reshape((424, 512))

# Define Kinect camera intrinsic parameters
fx = 364.9361572265625
fy = 364.9361572265625
cx = 257.80987548828125
cy = 207.2842559814453

# Compute 3D point coordinates from depth data
v, u = np.indices(depth_image.shape)
z = depth_image / 1000.0
x = (u - cx) * z / fx
y = (v - cy) * z / fy

# Flatten the 3D coordinates into a point cloud
point_cloud = np.stack((x.flatten(), y.flatten(), z.flatten()), axis=1)

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Visualize the point cloud
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(point_cloud[:,0], point_cloud[:,1], point_cloud[:,2])
plt.show()