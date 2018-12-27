import os
import torch
from torch.utils.data import Dataset
from torchvision.transforms.functional import to_tensor
from PIL import Image
from scipy.signal import convolve2d
import numpy as np
from DataInfoLoader import DataInfoLoader
import yaml
import math
from pathlib import Path
import cv2

def default_loader(path):
    return Image.open(path).convert('L')

def localNormalization(patch, P=3, Q=3, C=1):
    """将原始图片进行局部归一化
    Args:
        patch:输入为numpy 将Image转换成numpy格式

    Returns:
        patch_ln:局部归一化后的图片，类型为numpy
    """
    kernel = np.ones((P,Q)) / (P * Q)
    patch_mean = convolve2d(patch, kernel, boundary='symm', mode='same')
    patch_sm = convolve2d(np.square(patch), kernel, boundary='symm', mode='same')
    patch_std = np.sqrt(np.maximum(patch_sm - np.square(patch_mean), 0)) + C
    patch_ln = (patch - patch_mean)/patch_std
    return patch_ln

def patchSifting(im, patch_size=48, stride=48):
    """将原始图片进行局部归一化，然后挑选出有信息的部分
    Args:
        im: 原始图片，类型为numpy.ndarray
    
    Returns:
        patches:tensor类型(4维)

    """
    img=np.array(im).copy()
    im1=localNormalization(img)
    im1=Image.fromarray(im1)
    ret1,im2= cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    w, h = im1.size
    patches=()
    for i in range(0, h - stride, stride):
        for j in range(0, w - stride, stride):
            patch = im2[i:i+patch_size,j:j+patch_size]
            if judgeAllOnesOrAllZreos(patch)==False:
                patch=to_tensor(im1.crop((j,i,j+patch_size,i+patch_size)))
                patch=patch.float().unsqueeze(0)
                patches = patches + (patch,)
    return patches

def judgeAllOnesOrAllZreos(patch):
    flag1=True
    flag2=True
    for i in range(patch.shape[0]):
        for j in range(patch.shape[1]):
            if patch[i,j]==255:
                continue
            else:
                flag1=False

    for i in range(patch.shape[0]):
        for j in range(patch.shape[1]):
            if patch[i,j]==0:
                continue
            else:
                flag2=False
    return flag1 or flag2

class DIQADataset(Dataset):
    def __init__(self,dataset_name,config,data_index,status='train',loader=default_loader):
        self.loader = loader
        self.patch_size = 48
        self.stride = 48
        test_ratio = config['test_ratio']  
        train_ratio = config['train_ratio']
        dil=DataInfoLoader(dataset_name,config)
        img_name=dil.get_img_name()
        img_path=dil.get_img_path()
        qs_std=dil.get_qs_std()

        if status == 'train':
            self.index = data_index
            print("# Train Images: {}".format(len(self.index)))
        if status == 'test':
            self.index = data_index
            print("# Test Images: {}".format(len(self.index)))
        if status == 'val':
            self.index = data_index
            print("# Val Images: {}".format(len(self.index)))
        print('Index:')
        print(self.index)

        self.patches = ()
        self.label = []

        for idx in self.index:
            im=self.loader(img_path[idx])
            patches = patchSifting(im)
            if status == 'train':
                self.patches = self.patches + patches 
                for i in range(len(patches)):
                    self.label.append(qs_std[idx])
            else:
                self.patches = self.patches + (torch.stack(patches),)  
                self.label.append(qs_std[idx])

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        return (self.patches[idx], torch.Tensor([self.label[idx]]))

if __name__=='__main__':
    pass
    
    


    