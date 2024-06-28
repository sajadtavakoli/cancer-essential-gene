import math
import torch
import torchvision
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torchvision.transforms as T
from scipy.optimize import linear_sum_assignment
from torchvision.models._utils import IntermediateLayerGetter
from typing import Dict, List

from torch.nested import nested_tensor as NestedTensor
from collections import OrderedDict

from torch.nested import nested_tensor as NestedTensor
from collections import OrderedDict

import joblib

import torch
import torch.nn as nn
import torchvision
from torchsummary import summary

import torch.optim as optim
from torch.optim import lr_scheduler

from torch.utils.data import Dataset, DataLoader

import numpy as np 



class my_model(nn.Module):
    def __init__(self):
        super(my_model, self).__init__()
        #self.conv1 = nn.Conv2d(5, 16, kernel_size=(4, 30), stride=2, padding=2)
        #self.bn = nn.BatchNorm2d()
        #self.mp = nn.MaxPool2d()
        #self.ap = nn.AvgPool2d()
        conv1 = nn.Conv2d(5, 16, (4, 4), stride=(4, 1), padding=2)
        conv2 = nn.Conv2d(16, 32, (4, 4), stride=(4, 1), padding=2)
        conv3 = nn.Conv2d(32, 64, (4, 4), stride=(4, 1), padding=2)
        conv4 = nn.Conv2d(64, 64, (4, 4), stride=(4, 1), padding=2)
        conv5 = nn.Conv2d(64, 64, (2, 8), stride=(2, 1), padding=2)
        conv6 = nn.Conv2d(64, 128, (2, 2), stride=(2, 2), padding=1)
        conv7 = nn.Conv2d(128, 512, (2, 2), padding=2)
        
        self.out1 = nn.Sequential(nn.Conv2d(1, 16, (4, 4), stride=(1, 4), padding=2), # which one is x and which one is y
                                # nn.BatchNorm2d(num_features=16),
                                nn.ReLU()) # output size = (16, 8, (13112+4)/4=3279))


        self.out2 = nn.Sequential(nn.Conv2d(16, 32, (4, 4), stride=(1, 4), padding=2), 
                                  #nn.BatchNorm2d(32),
                                  nn.ReLU()) # output size = (32, 256, (3279+4)/4=820)
        
        self.out3 = nn.Sequential(nn.Conv2d(32, 64, (4, 4), stride=(1, 4), padding=2), # output size = (64, 8, (820+4)/4=206)
                                #nn.BatchNorm2d(64), 
                                nn.ReLU(),
                                nn.MaxPool2d((1, 2))) # output size = (64, 8, 103)

        self.out4 = nn.Sequential(nn.Conv2d(64, 128, (2, 2), stride=(1, 2), padding=2), # output size = (128, 8, 106=53)
                                #nn.BatchNorm2d(128), 
                                nn.ReLU(),
                                nn.MaxPool2d((1, 2))) # output size = (128, 8, 28)

        self.out5 = nn.Sequential(nn.Conv2d(128, 256, (2, 2), stride=(2, 2), padding=2), # output size = (512, /2, (28+4)/2=16) 
                                #nn.BatchNorm2d(512), 
                                nn.ReLU(),
                                nn.MaxPool2d((2, 2))) # output size =  (512, 4, 4)

        self.fc1 = torch.nn.Linear(7168, 512)
        self.fc2 = torch.nn.Linear(512, 64)
        self.fc3 = torch.nn.Linear(64, 1)

        """
        self.out6 = nn.Sequential(nn.Conv2d(64, 128, (2, 2), stride=(2, 2), padding=1), 
                                nn.BatchNorm2d(128), 
                                nn.ReLU(),
                                nn.MaxPool2d((2, 2)))
        

        self.out7 = nn.Sequential(nn.Conv2d(128, 512, (2, 2), padding=2), 
                                nn.BatchNorm2d(num_features=512),
                                nn.ReLU(), # output_size = (512, 10, 10) 
                                nn.MaxPool2d((2, 2)))
        """

    def forward(self, x):
        for lyr in [self.out1, self.out2, self.out3, self.out4, self.out5]:
            x = lyr(x)
        #x = x.view(x.size(0), -1)
        #print(x.shape)
        x = torch.flatten(x, start_dim=1)
        #print(x.shape)
        x = self.fc1(x)
        #print(x.shape)
        # x = torch.nn.Linear(x.shape[1], 512)(x)
        #x = torch.tanh(x)
        x = self.fc2(x)
        #x = torch.tanh(x)
        x = self.fc3(x)
        # x = torch.tanh(x)
        
        return x
    


# Custom dataset class
class Essentiality_Dataset(Dataset):
    def __init__(self, image_dir, annotation_dir):
        self.image_dir = image_dir
        #self.annotation_dir = annotation_dir
        self.labels = annotation_dir
        # self.image_files = [f for f in os.listdir(image_dir) if f.endswith('.npz')]
        # self.eps = 1e-8
        self.eps = 0
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        # print('\nidx', idx)
        img_path = self.image_dir[idx]
        img_name = img_path.split('/')[-1]
        key = img_name[:-4]
        #gene = img_name.split('-')[1]
        #cell = img_name.split('-')[2].split('.')[0]
        # print(gene, cell)

        # Normalizing expression using min-max method
        img = np.load(img_path)
        exp_mn, exp_mx, exp_mean = 2.8535, 15.1182, 6.677731137671191
        img[4] = (img[4]-exp_mean)/(exp_mx -exp_mn)
        img[5] = (img[5]-exp_mean)/(exp_mx -exp_mn)

        # Normalizing copy number variants using z-score method
        cop_mean, cop_std = 0.00912299710916001, 0.42795064034892055
        img[6] = (img[6]-cop_mean)/cop_std
        img[7] = (img[7]-cop_mean)/cop_std

        img = torch.from_numpy(img)
        img = img.reshape(shape=(1, img.shape[0], img.shape[1]))
        #print(img_name, gene, cell, key)
        ###
        # preprocessing and normalization
        ###
        # normalizing the label 
        mu, std = 0.002267139138399138, 0.2545539412173737
        lbl = self.labels[key]
        lbl = (lbl-mu)/std
        lbl = torch.as_tensor(lbl, dtype=torch.float32)

        #try:
        #    lbl = torch.as_tensor(self.labels[gene+'-'+cell], dtype=torch.float32)
        #except:
        #    print(img_name, gene, cell)

        return (img, lbl)
        



#def collate_fn(x):


# Paths to image and annotation directories
#train_imgs = '/work3/sajata/variant-calling/data/train'
#train_lbls = '/work3/sajata/variant-calling/data/train_label'
# train_imgs = '/work3/sajata/variant-calling/data/imgs_chr1'
# train_lbls = '/work3/sajata/variant-calling/data/imgs_chr1_label'

# val_imgs = '/work3/sajata/variant-calling/data/val'
# val_lbls = '/work3/sajata/variant-calling/data/val_label'
# val_imgs = '/work3/sajata/variant-calling/data/imgs_chr20'
# val_lbls = '/work3/sajata/variant-calling/data/imgs_chr20_label'

# test_imgs = '/work3/sajata/variant-calling/data/test'
# test_lbls = '/work3/sajata/variant-calling/data/test_label'


# dataset = VariantDataset(train_imgs, train_lbls)
# train_loader = DataLoader(dataset, batch_size=8, shuffle=True, collate_fn=lambda x: tuple(zip(*x)), num_workers=8)

# dataset = VariantDataset(val_imgs, val_lbls)
# val_loader = DataLoader(dataset, batch_size=8, shuffle=True, collate_fn=lambda x: tuple(zip(*x)), num_workers=8)

# dataset = VariantDataset(test_imgs, test_lbls)
# test_loader = DataLoader(dataset, batch_size=8, shuffle=True, collate_fn=lambda x: tuple(zip(*x)), num_workers=8)


if __name__=='__main__':
    temp_model = my_model()
    input_size = (1, 8, 13112)
    summary(temp_model, input_size)
    
