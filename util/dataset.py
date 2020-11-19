import os
import torch
import numpy as np
from plyfile import PlyData, PlyElement
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

category = ['Armchairs']#, 'Avoids']

def load_data_motion(seq_dir):
    list_input_data = []
    list_output_data = []
    list_seq_order = []
    list_seq_trans = []
    
    offset = 0
    for i in range(len(category)):
        filenames = os.listdir(os.path.join(seq_dir, category[i]))

        input_data = np.loadtxt(os.path.join(seq_dir, category[i], 'Input.txt'))
         
        output_data = np.loadtxt(os.path.join(seq_dir, category[i], 'Output.txt'))
         
        seq_order = np.loadtxt(os.path.join(seq_dir, category[i], 'Sequences.txt'), dtype=np.int32)
        seq_trans = np.loadtxt(os.path.join(seq_dir, category[i], 'ModelTrans.txt')).reshape((-1, 4, 4))

        seq_num = np.max(seq_order, axis=0)[0]
        list_input_data += ([None] * seq_num)
        list_output_data += ([None] * seq_num)
        list_seq_order += ([None] * seq_num)
        list_seq_trans += ([None] * seq_num)
        for k in range(seq_num):
            indices = (seq_order[:, 0] == k+1).squeeze()
            list_input_data[k + offset] = input_data[indices, :]
            list_output_data[k + offset] = output_data[indices, :]
            order = seq_order[indices, 1]
            list_seq_order[k + offset] = np.stack([np.ones(order.shape, dtype=np.int32)*i, order], axis=1)
            list_seq_trans[k + offset] = seq_trans[indices]
        offset += seq_num
    return list_seq_order, list_input_data, list_output_data, list_seq_trans

def load_label_dict(dir):
    label_dict = {}
    motion_label = open(dir)
    count = 0
    for line in motion_label:
        label_dict[line.split()[1]] = count
        count += 1 
    motion_label.close()
    return label_dict

def load_contacts(dir, bone2id):
    list_contacts = []
    offset = 0
    for i in range(len(category)):
        file = open(os.path.join(dir, category[i], 'Contacts.txt'))
        lines = file.readlines()
        contacts = []
        c = []
        for line in lines:
            if line == "\n":
                contacts.append(c)
                c = []
                continue
            bone = line.split()[0]
            c.append([int(bone2id[bone]), float(line.split()[1]), float(line.split()[2]), float(line.split()[3])]) 
        contacts = np.array(contacts)   
        file.close()
        list_contacts.append(contacts)

    return list_contacts


class MotionSeqDataset(Dataset):
    def __init__(self, seq_dir, label_dict_dir):
        """
        Args:
            seq_path(string): Path to the motion sequence file.
        """
        self.seq2model , self.list_input_data, self.list_gt_data, self.list_seq_trans = load_data_motion(seq_dir)
        self.inputlabel_dict = load_label_dict(os.path.join(label_dict_dir, 'InputLabels.txt'))
        self.gtlabel_dict = load_label_dict(os.path.join(label_dict_dir, 'OutputLabels.txt'))
        self.bone2id = {'Hips':1, 'Chest':2, 'Chest2':3, 'Chest3':4, 'Chest4':5, 'Neck':6, 'Head':7, 
                        'RightCollar':8, 'RightShoulder':9, 'RightElbow':10, 'RightWrist':11, 
                        'LeftCollar':12, 'LeftShoulder':13, 'LeftElbow':14, 'LeftWrist':15, 
                        'RightHip':16, 'RightKnee':17, 'RightAnkle':18, 'RightToe':19, 
                        'LeftHip':20, 'LeftKnee':21, 'LeftAnkle':22, 'LeftToe':23}
        self.joints_num = len(self.bone2id)

        self.input_size = len(self.inputlabel_dict) - 16 + self.joints_num * 3
        self.gt_size = len(self.gtlabel_dict)

        self.list_contact = load_contacts(seq_dir, self.bone2id)
       
    def __len__(self):
        return len(self.list_input_data)

    def __getitem__(self, idx):
        return self.seq2model[idx], self.list_input_data[idx][:, 16:], self.list_gt_data[idx], self.list_seq_trans[idx]

    def getFrameNumber(self, idx):
        return self.list_input_data[idx].shape[0]

    def ExtractJointLocalTrans(self, idx, frame):
        seq = self.list_input_data[idx]
        trans = np.zeros(shape=[len(self.bone2id), 4, 4])
        for b_name, b_id in self.bone2id.items():
            for i in range(4):
                for j in range(4):
                    trans[b_id - 1, i, j] = seq[frame, self.inputlabel_dict['Bone' + str(b_id) + b_name +str(i)+str(j)]]
            # positions[:, b_id - 1, 0] = seq[:, self.label_dict['Bone' + str(b_id) + b_name + 'PositionX']]
            # positions[:, b_id - 1, 1] = seq[:, self.label_dict['Bone' + str(b_id) + b_name + 'PositionY']]
            # positions[:, b_id - 1, 2] = seq[:, self.label_dict['Bone' + str(b_id) + b_name + 'PositionZ']]
        return trans

    def ExtractRootsTransform(self, idx, frame):
        seq = self.list_input_data[idx]
        roots = np.zeros(shape=[4, 4])
        for i in range(4):
            for j in range(4):
                roots[i, j] = seq[frame, self.inputlabel_dict['Root'+str(i)+str(j)]]
        return roots

    def ExtractJointGlobalPosition(self, idx, frame):
        local_trans = self.ExtractJointLocalTrans(idx, frame)
        roots = self.ExtractRootsTransform(idx, frame)
        
        position = np.zeros(shape=[len(self.bone2id), 3])
        for b_id in range(len(self.bone2id)):
            trans = np.matmul(roots, local_trans[b_id, :, :])
            position[b_id, :] =  np.matmul(trans, np.array([0, 0, 0, 1]))[:3]

        return position






def load_data_model(model_path, pts_num, dim):
    model_count = 0
    model_offset = []
    for i in range(len(category)):
        filenames = os.listdir(os.path.join(model_path, category[i]))
        model_offset.append(model_count) 
        model_count += len(filenames)
        
 
    model_data = np.zeros(shape=[model_count, pts_num, dim], dtype=np.float32)
     
    for i in range(len(category)):
        filenames = os.listdir(os.path.join(model_path, category[i]))
        for j in range(len(filenames)):
            data = PlyData.read(os.path.join(model_path, category[i], str(j) + '.ply'))
            model_data[j + model_offset[i], :, 0] = data['vertex'].data['x'][:pts_num]
            model_data[j + model_offset[i], :, 1] = data['vertex'].data['y'][:pts_num]
            model_data[j + model_offset[i], :, 2] = data['vertex'].data['z'][:pts_num]

    return model_data, model_offset
    

class ShapeDataset(Dataset):
    def __init__(self, model_path, model_class, pts_num=2048, dim=3):
        """
        Args:
            model_path(string): Path to the model files.
            pts_num(int): The number of sampled points.
            dim(int): Dimension of points.
        """
        self.model_data, self.model_offset = load_data_model(model_path, pts_num, dim)
        self.cat2id = {'airplane': 0, 'bag': 1, 'cap': 2, 'car': 3, 'chair': 4, 
                       'earphone': 5, 'guitar': 6, 'knife': 7, 'lamp': 8, 'laptop': 9, 
                       'motor': 10, 'mug': 11, 'pistol': 12, 'rocket': 13, 'skateboard': 14, 'table': 15}
        self.seg_num = [4, 2, 2, 4, 4, 3, 3, 2, 4, 2, 6, 2, 3, 3, 3, 3]
        self.index_start = [0, 4, 6, 8, 12, 16, 19, 22, 24, 28, 30, 36, 38, 41, 44, 47]
        self.pts_num = pts_num    
        self.model_class = model_class

        id_choice = self.cat2id[self.model_class]
        self.seg_num_all = self.seg_num[id_choice]
        self.seg_start_index = self.index_start[id_choice]
        self.model_label = np.ones(self.model_data.shape[0], dtype=np.int32) * id_choice

    def __len__(self):
        return self.model_data.shape[0]

    def __getitem__(self, item):
        model_label = self.model_label[item]
        data = self.model_data[item, :self.pts_num, :]
        return data, model_label

    def getID(self, motion_class, id_in_class):
        return self.model_offset[motion_class] + id_in_class


if __name__ == "__main__":
    motiondata = MotionSeqDataset(seq_dir='./data/motion/', label_dict_dir='./data/motion/')
    print(motiondata.list_contact[0][0])
                

        
