# 测试模型top1的准确率
from __future__ import print_function
import pathlib
import os
import sys
import argparse
import time
import math
import numpy as np
from PIL import Image
import pandas as pd
# import tensorboard_logger as tb_logger
import torch
import torch.backends.cudnn as cudnn
from torchvision import transforms, datasets
from torch.utils.data import Dataset
##from util import adjust_learning_rate, warmup_learning_rate
#from util import set_optimizer, save_model
#from networks.resnet_big import SupConResNet
#from losses import SupConLoss
import torch.nn.functional as F
from arcface import Arcface

try:
    import apex
    from apex import amp, optimizers
except ImportError:
    pass

#os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, [1, 2]))


device      = torch.device('cpu')
# load the template1

def parse_option():
    parser=argparse.ArgumentParser("argument for generating csv!")
    parser.add_argument("--batch_size",type=int ,default=20)
    parser.add_argument('--num_workers',type=int,default=16)
    parser.add_argument("--model",type=str,default="resnet18")
    parser.add_argument('--dataset',type=str,default='myset')
    parser.add_argument("--test_folder",default='/data/arcface-pytorch-main/test.txt',type=str)
    opt=parser.parse_args()
    return opt


model = Arcface()

class myDataset(Dataset):
    def __init__(self, datafile, transform=None, target_transform=None):

        self.transform = transform
        self.target_transform = target_transform

        data = pd.read_csv(datafile)
        self.data = list(data['paths'].values)
        self.targets = list(data['labels'].values)
        self.classes = np.arange(len(set(self.targets)))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        path, label = self.data[idx], self.targets[idx]
        image = Image.open(path)
        if self.transform:
            image = self.transform(image)
        return image, label


def compare_template_image(template,image):
    distances=[100 for i in range(template.size()[0])]
    for j in range(template.size()[0]):
        for k in range(template.size()[1]):
            distances[j]=min(float(F.pairwise_distance(image,template[j][k],p=2)),distances[j])
    return distances.index(min(distances))



def compute_acc1(opt,template):
    f1 = 0
    all = 0
    for (image,label) in  myDataset(opt.test_folder):
        feature = model.get_features(image)
        feature = torch.tensor(feature)
        y = compare_template_image(template,feature)
        if y == label:
            f1 = f1+1
            all = all+1
        else:
            all = all+1
            print(y,label)
    acc = f1/all
    return acc

def compute_acc2(opt,weight):
    f1 = 0
    all = 0
    for (image,label) in  myDataset(opt.test_folder):
        feature = model.get_features(image)
        y = compare_weight_image(weight,feature)
        if label == y:
            f1 = f1+1
            all = all+1
        else:
            all = all+1
            print(y,label)
    acc = f1/all
    return acc


def load_csv(csv_path):
    template = []
    csv_list = [csv_path]
    csv_list = sorted(csv_list)
    for csv_file in csv_list:
        #print(str(csv_file))
        df = pd.read_csv(str(csv_file), header=None)
        data = df.values
        data = data.astype(float)
        data = torch.from_numpy(data)
        template.append(data)
    return torch.stack(template, dim=0)


def load_template(template_path):
    template = []
    csv_list = [str(csv_file) for csv_file in pathlib.Path(template_path).glob("*.csv")]
    csv_list = sorted(csv_list)
    for csv_file in csv_list:
        #print(str(csv_file))
        df = pd.read_csv(str(csv_file), header=None)
        data = df.values
        data = data.astype(float)
        data = torch.from_numpy(data)
        template.append(data)
    return torch.stack(template, dim=0)


# compare the image to the template1
def compare_weight_image(weight, image):
    distances = [100 for i in range(553)]
    x1 = image
    for i in range (553):
        x2 = weight[0][i]
        distances[i] = min(float(np.dot(x1,x2)/(np.linalg.norm(x1) * np.linalg.norm(x2))),distances[i])
    label1 = distances.index(max(distances))
    label2 = distances.index(sorted(distances)[-2])
    label3 = distances.index(sorted(distances)[-3])
    label4 = distances.index(sorted(distances)[-4])
    label5 = distances.index(sorted(distances)[-5])
    label = [label1, label2, label3, label4, label5]
    return label1

def main():
    opt = parse_option()
    #template = load_template("/data/arcface-pytorch-main/csv")
    weight = load_csv("/data/arcface-pytorch-main/weight/iresnet18.csv")
    print("==========>testing")

    with torch.no_grad():
         #acc1 = compute_acc1(opt,template)
         acc2 = compute_acc2(opt,weight)
         print(">>>>>>>>>>>>>>>>>>>>the accuracy:",acc2)
         #print(">>>>>>>>>>>>>>>>>>>>the label:{label}".format(label=label))


if __name__ == '__main__':
    start = time.time()
    main()
    end = time.time()
    print("测试时间:",end-start)
