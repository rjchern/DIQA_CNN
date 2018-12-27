from PIL import Image
import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
from torchvision.transforms.functional import to_tensor
from torch.autograd import Variable as V 
from scipy.signal import convolve2d
import cv2

class CNNDIQAnet(nn.Module):
    def __init__(self):
        super(CNNDIQAnet, self).__init__()
        self.conv1 = nn.Conv2d(1, 40, 5)
        self.pool1 = nn.MaxPool2d(4)
        self.conv2 = nn.Conv2d(40, 80, 5)
        self.fc1   = nn.Linear(160, 1024)
        self.fc2   = nn.Linear(1024, 1024)
        self.fc3 = nn.Linear(1024, 1)
    
    def forward(self, x):
        x=x.view(-1,x.size(-3),x.size(-2),x.size(-1))
        x  = self.conv1(x)
        x  = self.pool1(x)
        x  = self.conv2(x)
        x1 = F.max_pool2d(x, (x.size(-2), x.size(-1)))
        x2 = -F.max_pool2d(-x, (x.size(-2), x.size(-1)))
        x  = torch.cat((x1, x2), 1)  
        x  = x.squeeze(3).squeeze(2)
        x  = F.relu(self.fc1(x))
        x  = F.dropout(x)
        x  = F.relu(self.fc2(x))
        x  = self.fc3(x)
        return x

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

class Solver:
    def __init__(self):
        #pre-trained model path
        self.model_path='./checkpoints/CNNDIQA-SOC-EXP916-lr=0.0001.pth'
        #Initialize the model
        self.model=CNNDIQAnet()

    def quality_assessment(self,img_path):
        #load the pre-trained model
        self.model.load_state_dict(torch.load(self.model_path))
        im=Image.open(img_path).convert('L')
        im=torch.stack(patchSifting(im))
        #im=to_tensor(im).unsqueeze(0)
        # if torch.cuda.is_available():
        #     model=model.cuda()
        #     im=im.cuda()
        qs=self.model(im)
        qs=qs.data.squeeze(0).cpu().numpy()[:,0].mean()
        return qs

if __name__=='__main__':
    #Example
    img_path='/media/crj/Earth/Dataset/DIQA_Dataset/SOC/Images/2012-04-16_17-32-27_295.jpg'
    solver=Solver()
    qs=solver.quality_assessment(img_path)
    print('quality score is:{}'.format(qs))