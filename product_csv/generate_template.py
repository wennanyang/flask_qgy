import sys
import os
import argparse
import time
import math
import numpy as np
from PIL import Image
import pandas as pd
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import Dataset
from torchvision import transforms
from SupContrast.networks.resnet_big import  SupConResNet


os.environ["CUDA_VISIBLE_DEVICES"]='3,2,0,1'.join(map(str, [0,1,2,3]))

def parse_option():
    parser=argparse.ArgumentParser("argument for generating csv!")
    parser.add_argument("--batch_size",type=int ,default=20)
    parser.add_argument('--num_workers',type=int,default=16)

    parser.add_argument("--model",type=str,default="resnet18")
    parser.add_argument('--dataset',type=str,default='myset')

    parser.add_argument("--ckpt",type=str,default='/data/codes/resnet/models/SupCon_myset_resnet18_bsz_100_trial_0/last.pth')
    parser.add_argument("--data_folder",type=str,default='/data/codes/flask_product/product_csv/')

    opt=parser.parse_args()
    return opt

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

def set_loader(opt):
    mean = (0.,0.,0.)
    std = (1.,1.,1.)
    normalize = transforms.Normalize(mean=mean, std=std)
    val_transform = transforms.Compose([
        transforms.Resize((128,128)),
        transforms.ToTensor(),
        normalize,
    ])
    val_dataset = myDataset(opt.data_folder+'templates.csv', transform=val_transform)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=opt.batch_size, shuffle=False,
        num_workers=opt.num_workers, pin_memory=True)
    return val_loader

def set_model(opt):
    model = SupConResNet(name=opt.model)

    ckpt = torch.load(opt.ckpt, map_location='cpu')
    state_dict = ckpt['model']

    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            model.encoder = torch.nn.DataParallel(model.encoder)
        else:
            new_state_dict = {}
            for k, v in state_dict.items():
                k = k.replace("module.", "")
                new_state_dict[k] = v
            state_dict = new_state_dict
        model = model.cuda()
        cudnn.benchmark = True

        model.load_state_dict(state_dict)

    return model

def embedding(train_loader, model, opt):
    map_embeddings = {}
    for (images, labels) in train_loader:
        images = images.cuda(non_blocking=True)
        embeddings = model(images).detach().cpu().numpy()
        labels = labels.detach().cpu().numpy()
        
        for embedding, label in zip(embeddings, labels):
            if label not in map_embeddings:
                map_embeddings[label] = [embedding]
            else:
                map_embeddings[label].append(embedding)
    csv_path = 'features'
    if not os.path.exists(csv_path):
        os.mkdir(csv_path)
    for label,emb in map_embeddings.items():
        emb = np.array(emb)
        name = []
        for l in range(emb.shape[1]):
            name.append('x'+str(l))
        df = pd.DataFrame(emb,columns=name)
        labels = [label]*len(emb)
        #df['labels'] = labels
        #df = df[['labels',*name]]
        fpath = os.path.join(csv_path,f'{str(label)}.csv')
        df.to_csv(fpath ,index=False,header=None)
    return map_embeddings




if __name__=="__main__":
    opt=parse_option()
    val_loader=set_loader(opt)
    model=set_model(opt)
    model.eval()
    with torch.no_grad():
        map_embeddings = embedding(val_loader, model, opt)


