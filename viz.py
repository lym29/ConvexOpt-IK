"""Functions to visualize human poses"""
# https://github.com/una-dinosauria/human-motion-prediction/blob/master/src/viz.py
# adapt xyz position array data and show in 3D axis

import matplotlib.pyplot as plt
import numpy as np
import h5py
import os
from mpl_toolkits.mplot3d import Axes3D

class Ax3DPose(object):
  def __init__(self, ax, start, end, lcolor="#3498db", rcolor="#e74c3c"):
    """
    Create a 3d pose visualizer that can be updated with new poses.
    Args
      ax: 3d axis to plot the 3d pose on
      lcolor: String. Colour for the left part of the body
      rcolor: String. Colour for the right part of the body
    """

    # Start and endpoints of our representation ---bones of skeleton
    #self.I   = np.array([1,2,3,4,5,6,8, 9,10,12,13,14,16,17,18,20,21,22,5, 5, 1, 1])-1
    #self.J   = np.array([2,3,4,5,6,7,9,10,11,13,14,15,17,18,19,21,22,23,8,12,16,20])-1
    self.I = start - 1
    self.J = end - 1
    # Left / right indicator
    # self.LR  = np.array([1,1,1,0,0,0,0, 0, 0, 0, 0, 0, 0, 1, 1, 1], dtype=bool)
    self.ax = ax

    vals = np.zeros((23, 3)) # skeleton vals(joints,xyz)

    model = np.zeros(shape=[3, 100])
    self.pts_plot = self.ax.plot(model[0, :], model[1, :], model[2, :], c=rcolor, marker='o', linestyle = 'None')

    descriptor = np.zeros(shape=[3, 3])
    self.descriptor_plot = self.ax.plot(descriptor[:, 0], model[:, 1], model[:, 2], c='g', marker='o', linestyle = 'None')

    # Make connection matrix
    self.plots = []
    for i in np.arange( len(self.I) ):
      x = np.array( [vals[self.I[i], 0], vals[self.J[i], 0]] )
      y = np.array( [vals[self.I[i], 1], vals[self.J[i], 1]] )
      z = np.array( [vals[self.I[i], 2], vals[self.J[i], 2]] )
      self.plots.append(self.ax.plot(x, y, z, lw=2, c=lcolor))

    self.ax.set_xlabel("x")
    self.ax.set_ylabel("y")
    self.ax.set_zlabel("z")

    self.ax.set_xlim3d([-5, 5])
    self.ax.set_zlim3d([-1.5, 3])
    self.ax.set_ylim3d([0, 2])
  
  # refresh frame by frame of the pose
  def update(self, vals, model, descriptors, lcolor="#3498db", rcolor="#e74c3c"):
    """
    Update the plotted 3d pose.
    Args
      channels: 23*3=69-dim long np array. The pose to plot.
      lcolor: String. Colour for the left part of the body.
      rcolor: String. Colour for the right part of the body.
    Returns
      Nothing. Simply updates the axis with the new pose.
    """
    for i in np.arange( len(self.I) ):
      x = np.array( [vals[self.I[i], 0], vals[self.J[i], 0]] )
      y = np.array( [vals[self.I[i], 1], vals[self.J[i], 1]] )
      z = np.array( [vals[self.I[i], 2], vals[self.J[i], 2]] )
      self.plots[i][0].set_xdata(x)
      self.plots[i][0].set_ydata(z)
      self.plots[i][0].set_3d_properties(y)
      self.plots[i][0].set_color(lcolor)

    self.pts_plot[0].set_xdata(model[0, :])
    self.pts_plot[0].set_ydata(model[2, :])
    self.pts_plot[0].set_3d_properties(model[1, :])

    self.descriptor_plot[0].set_xdata(descriptors[:, 0])
    self.descriptor_plot[0].set_ydata(descriptors[:, 2])
    self.descriptor_plot[0].set_3d_properties(descriptors[:, 1])
    
    model_min = np.min(model, axis=1)
    model_max = np.max(model, axis=1)
    model_center = (model_min + model_max)/2
    
    r = 1
    xroot, yroot, zroot = vals[0,0], vals[0,1], vals[0,2]
    #print(xroot, yroot, zroot)
    self.ax.set_xlim3d([-r+xroot, r+xroot])
    self.ax.set_ylim3d([-r+zroot, r+zroot])
    self.ax.set_zlim3d([-r+yroot, r+yroot])
    #self.ax.set_ylim3d([-1, 2])
    # self.ax.set_xlim3d([-r+model_center[0], r+model_center[0]])
    # self.ax.set_zlim3d([-r+model_center[1], r+model_center[1]])
    # self.ax.set_ylim3d([-r+model_center[2], r+model_center[2]])

    self.ax.set_aspect('auto')
