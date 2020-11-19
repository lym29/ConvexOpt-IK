from __future__ import division

import numpy as np
import h5py
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
import viz
import time
import copy
import scipy.io
from util.dataset import ShapeDataset, MotionSeqDataset

JOINTS_NUM = 23

def main():
    def read_mat(mat_path):
        mat=scipy.io.loadmat(mat_path)
        #a=mat['position'].transpose(2,1,0)
        a=mat['position']
        return a # ndarray type

    seq_id = 0
    motiondataset = MotionSeqDataset(seq_dir='./data/motion/', label_dict_dir='./data/motion/')
    seq2model, _, _, model_trans = motiondataset[seq_id]
    #roots = motiondataset.ExtractRoots(seq_id)
    b2id = motiondataset.bone2id
    start = np.array([b2id['Hips'], b2id['Chest'], b2id['Chest2'], b2id['Chest3'], b2id['Chest4'], b2id['Neck'], 
            b2id['RightCollar'], b2id['RightShoulder'], b2id['RightElbow'], 
            b2id['LeftCollar'], b2id['LeftShoulder'], b2id['LeftElbow'],  
            b2id['RightHip'], b2id['RightKnee'], b2id['RightAnkle'], 
            b2id['LeftHip'], b2id['LeftKnee'], b2id['LeftAnkle'],
            b2id['Chest4'], b2id['Chest4'], b2id['Hips'], b2id['Hips']])
    end = np.array([b2id['Chest'], b2id['Chest2'], b2id['Chest3'], b2id['Chest4'], b2id['Neck'], b2id['Head'],
            b2id['RightShoulder'], b2id['RightElbow'], b2id['RightWrist'],
            b2id['LeftShoulder'], b2id['LeftElbow'], b2id['LeftWrist'],
            b2id['RightKnee'], b2id['RightAnkle'], b2id['RightToe'],
            b2id['LeftKnee'], b2id['LeftAnkle'], b2id['LeftToe'],
            b2id['RightCollar'], b2id['LeftCollar'], b2id['RightHip'], b2id['LeftHip']])

    # print(start)
    # print(end)
    
    frame_number = motiondataset.getFrameNumber(seq_id) # get frame number of current file

    
    modeldataset = ShapeDataset(model_path='./data/pointscloud/', model_class='chair', pts_num = 1024)

    # === Plot and animate ===

    fig = plt.figure()
    ax = plt.gca(projection='3d')
    ob = viz.Ax3DPose(ax, start, end)

    contacts = motiondataset.list_contact[seq_id]


    # Plot the conditioning ground truth
    '''int(frame_number/2)'''
    for i in range(0, frame_number):
        mid = modeldataset.getID(seq2model[i, 0], seq2model[i, 1])
        print(mid)
        # if mid != 0 and mid != 12:
        #     continue
        # if mid == 12:
        #     print(model_trans[i])
        # if mid == 0:
        #     print(model_trans[i])
        
        # print(motiondataset.ExtractJointGlobalPosition(seq_id, i)[0])
        ptcloud, _ = modeldataset[mid]
        ptcloud = ptcloud.T
        #ptcloud = ptcloud - np.mean(ptcloud, axis = 1, keepdims=True)
        temp = np.concatenate((ptcloud, np.ones(shape=[1, ptcloud.shape[1]])), axis=0)
        temp = np.matmul(model_trans[i], temp)
        ptcloud = temp[:3, :] #+ np.expand_dims(roots[i, :].T, axis=1)
        ob.update(motiondataset.ExtractJointGlobalPosition(seq_id, i), ptcloud, contacts[i, :, 1:])
        plt.show(block=False)
        fig.canvas.draw()
        plt.pause(0.01)


if __name__ == '__main__':
    main()
