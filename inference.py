#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn
import torch.nn.functional as F
from natsort import natsorted
import numpy as np
#import config
import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import save_image
import torchvision.transforms as transforms
import cv2
import torch.nn as nn
import torch.nn.functional as F
import torchvision

#use this class for training

class MapDataset(Dataset):
    def __init__(self, root_dir,target_dir,transform=None):
        self.root_dir = root_dir
        self.target_dir=target_dir
        self.list_files_x = os.listdir(self.root_dir)
        self.list_files_y = os.listdir(self.target_dir)
        self.transform=transform

    def __len__(self):
        return len(self.list_files_x)

    def __getitem__(self, index):
        img_file_x= self.list_files_x[index]
        img_file_y= self.list_files_y[index]
        img_path_x = os.path.join(self.root_dir, img_file_x)
        img_path_y = os.path.join(self.target_dir,img_file_x)
        input_image=  cv2.imread(img_path_x)
        target_image=  cv2.imread(img_path_y)
        
        input_image1=cv2.cvtColor(input_image,cv2.COLOR_BGR2GRAY)
        target_image1=cv2.cvtColor(target_image,cv2.COLOR_BGR2GRAY)
        
        input_image = self.transform(input_image1)
        target_image= self.transform(target_image1)
        
        mean_input,std_input=input_image.mean([1,2]),input_image.std([1,2])
        #mean_target,std_target=target_image.mean([1,2]),target_image.std([1,2])
        
        input_image=(input_image-mean_input)/(std_input)
        #target_image=(target_image-mean_target)/(std_target)
        
        return input_image,target_image


# In[2]:



import numpy as np
#import config
import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import save_image
import torchvision.transforms as transforms
import cv2
import torch.nn as nn
import torch.nn.functional as F
import torchvision

#use this class for inference or validation

class TestDataset(Dataset):
    def __init__(self, root_dir,transform=None):
        self.root_dir = root_dir
        #self.target_dir=target_dir
        self.list_files_x = natsorted(os.listdir(self.root_dir))
        #self.list_files_y = os.listdir(self.target_dir)
        self.transform=transform

    def __len__(self):
        return len(self.list_files_x)

    def __getitem__(self, index):
        img_file_x= self.list_files_x[index]
        #img_file_y= self.list_files_y[index]
        img_path_x = os.path.join(self.root_dir, img_file_x)
        #img_path_y = os.path.join(self.target_dir,img_file_x)
        input_image=  cv2.imread(img_path_x)
        #target_image=  cv2.imread(img_path_y)
        
        input_image1=cv2.cvtColor(input_image,cv2.COLOR_BGR2GRAY)
        #target_image1=cv2.cvtColor(target_image,cv2.COLOR_BGR2GRAY)
        
        input_image = self.transform(input_image1)
        #target_image= self.transform(target_image1)
        
        mean_input,std_input=input_image.mean([1,2]),input_image.std([1,2])
        #mean_target,std_target=target_image.mean([1,2]),target_image.std([1,2])
        
        input_image=(input_image-mean_input)/(std_input)
        #target_image=(target_image-mean_target)/(std_target)
        
        return input_image


# In[4]:


dataset=TestDataset("/path/to/inference/dataset")
                   transform=transforms.Compose([transforms.ToTensor()]))
testloader=DataLoader(dataset,batch_size=1,shuffle=True)
#x,y=next(iter(trainloader)


# In[5]:


class Down_conv(nn.Module):
    def __init__(self,in_ch,out_ch,k=3,s=2,pad=1):
        super().__init__()
        self.down_sample=nn.Conv2d(in_channels=in_ch,out_channels=out_ch,kernel_size=k,stride=s,padding=pad)
        self.relu=nn.ReLU()

    def forward(self,x):
        return self.relu(self.down_sample(x))

        

class Flat_conv(nn.Module):
    def __init__(self,in_ch,out_ch,k=3,s=1,pad=1):
        super().__init__()
        self.flat_conv=nn.Conv2d(in_channels=in_ch,out_channels=out_ch,kernel_size=k,stride=s,padding=pad)
        self.relu=nn.ReLU()
    
    def forward(self,x):
        return self.relu(self.flat_conv(x))


class Up_conv(nn.Module):
    def __init__(self,scale):
        super().__init__()
        #self.upconv=nn.ConvTranspose2d(in_channels=in_ch,out_channels=out_ch,kernel_size=4,stride=(0.5,0.5))
        self.upconv=nn.Upsample(scale_factor=scale,mode="nearest")
        self.relu=nn.ReLU()
        
    def forward(self,x):
        return self.relu(self.upconv(x))


# In[6]:


class Encoder2(nn.Module):
    def __init__(self):
        super().__init__()
        self.block1=nn.Sequential(Down_conv(1,32),Flat_conv(32,64),Flat_conv(64,64))
        self.block2=nn.Sequential(Down_conv(64,128),Flat_conv(128,128),Flat_conv(128,128))
        self.block3=nn.Sequential(Down_conv(128,256),Flat_conv(256,256),
                                  Flat_conv(256,256))
        self.block4=nn.Sequential(Down_conv(256,512),Flat_conv(512,512),Flat_conv(512,512))

class Decoder2(nn.Module):
    def __init__(self):
        super().__init__()
        self.dec_block1=nn.Sequential(Up_conv(scale=2),Flat_conv(512,256),Flat_conv(256,256))
        self.dec_block2=nn.Sequential(Up_conv(scale=2),Flat_conv(512,256),Flat_conv(256,128))
        self.dec_block3=nn.Sequential(Up_conv(scale=2),Flat_conv(256,128),Flat_conv(128,64))
        self.dec_block4=nn.Sequential(Up_conv(scale=2),Flat_conv(128,64),Flat_conv(64,32))


# In[7]:


class Own_arch(Encoder2,Decoder2):
    def __init__(self):
        super().__init__()
        #self.enc=Encoder()
        #self.dec=Decoder()
        self.final=nn.Conv2d(32,1,kernel_size=3,stride=1,padding=1)
        
    def forward(self,x):
        block1_out=self.block1(x)
        block2_out=self.block2(block1_out)
        block3_out=self.block3(block2_out)
        block4_out=self.block4(block3_out)
        dec_out1=self.dec_block1(block4_out)
        concat1=torch.cat([dec_out1,block3_out],axis=1)
        dec_out2=self.dec_block2(concat1)
        concat2=torch.cat([dec_out2,block2_out],axis=1)
        dec_out3=self.dec_block3(concat2)
        concat3=torch.cat([dec_out3,block1_out],axis=1)
        dec_out4=self.dec_block4(concat3)
        return self.final(dec_out4)


# In[8]:


model=Own_arch()

model.load_state_dict(torch.load("epch_400.pth")) #path of pretrained model weights

model.eval()

