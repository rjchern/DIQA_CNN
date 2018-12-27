import torch
from torch import nn
import torch.nn.functional as F
from torchvision.transforms import ToTensor,ToPILImage
from torch.autograd import Variable as V 
from PIL import Image
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

if __name__=='__main__':
    pass