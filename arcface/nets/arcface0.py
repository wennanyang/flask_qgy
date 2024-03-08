import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Module, Parameter
from nets.resnet import resnet18
from nets.iresnet import iresnet18, iresnet50, iresnet34, iresnet200, iresnet100
from nets.mobilefacenet import get_mbf
from nets.mobilenet import get_mobilenet
from nets.vit_face import vit_face
from convformer.models.models import get_model as gm
from backbones import get_model
from backbones.resnet_vit import VisionTransformer as resnet_vit
from backbones.fpn_vit import VisionTransformer as fpn_vit
from backbones.BoTNet import  ResNet50
from backbones.nextvit import nextvit_base
from nets.resnet import resnet18
from backbones.vit_face import ViT
from backbones.conformer.vision_transformer import VisionTransformer
from backbones.swin_transformer import SwinTransformer
from backbones.conformer.models import Conformer_small_patch16
from backbones.vit_pytorch.t2t import T2TViT
from backbones.vit_pytorch1.vit.vit import VisionTransformer as vit
#from convit.convit import VisionTransformer
from backbones.dilate import Dilateformer
from backbones.dilate2 import Dilateformer2
from backbones.dilate3 import Dilateformer3
from backbones.resnest.torch.models.resnest import resnest50 as resnest


class SubArcface_Head(nn.Module):
    def __init__(
        self,
        in_features=128,
        out_features=9691,
        scale=64.,
        margin=0.5,
        subcenters=5,
        with_theta=True,
        clip_thresh=True,
        clip_value=True,
        with_weight=True,
        fc_std=True,
        if_clip=False,
    ):
        super(SubArcface_Head, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.scale = scale
        self.margin = margin
        self.K = subcenters
        self.with_theta = with_theta
        self.clip_thresh = clip_thresh
        self.clip_value = clip_value
        self.with_weight = with_weight
        self.if_clip = if_clip
        self.pool = nn.MaxPool1d(self.K)
        self.weight = Parameter(torch.Tensor(out_features,self.K, in_features))
        # self.reset_parameters()
        self.weight.data.normal_(std=fc_std)

        self.thresh = math.cos(math.pi - self.margin)
        self.mm = math.sin(math.pi - self.margin) * self.margin
        self.cnt = 0

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.weight.size(1))
        # self.logits.weight.data.normal_(std=self.config["fc_std"])
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, input, label):
        ex = input / torch.norm(input, 2, 1, keepdim=True)
        ew = self.weight / torch.norm(self.weight, 2, 2, keepdim=True)
        cos = torch.matmul(ew,ex.t())
        cos = cos.permute(2,0,1)
        if self.with_theta:
            non_pool_cos = torch.clone(cos)
        cos = self.pool(cos).squeeze()

        index = torch.where(label!=-1)[0]
        if self.if_clip:
            a = torch.zeros_like(cos)
            b = torch.zeros_like(cos)
            a.scatter_(1,label[index,None],self.margin)
            b.scatter_(1,label[index,None],-self.mm)
            mask = (cos>self.thresh)*1
            logits =  torch.cos(cos.acos_() + a * mask) + b * ( 1 - mask )
            logits = self.scale * logits
        else:
            m_hot = torch.zeros_like(cos)
            m_hot.scatter_(1,label[index,None],self.margin)
            cos.acos_()
            cos[index] += m_hot
            cos.cos_().mul_(self.scale)
            logits = cos
        return  logits

class Arcface_Head(Module):
    def __init__(self, embedding_size=128, num_classes=553, s=64., m=0.5):
        super(Arcface_Head, self).__init__()
        self.s = s
        self.m = m
        self.weight = Parameter(torch.FloatTensor(num_classes, embedding_size))
        # self.weight = torch.from_numpy(np.load(weight))
        nn.init.xavier_uniform_(self.weight)

        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m


    def forward(self, input, label):
        cosine = F.linear(input, F.normalize(self.weight))
        # cosine = F.linear(input, self.weight)
        #w = F.normalize(self.weight).detach().cpu().numpy()
        #np.savetxt('weight/dilate3.csv', w, delimiter=',')
        sine = torch.sqrt((1.0 - torch.pow(cosine, 2)).clamp(0, 1))
        phi = cosine * self.cos_m - sine * self.sin_m
        phi = torch.where(cosine.float() > self.th, phi.float(), cosine.float() - self.mm)
        cosine = cosine
        one_hot = torch.zeros(cosine.size()).type_as(phi).long()
        #print(one_hot)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s
        # print(w)
        return output


class Arcface(nn.Module):
    def __init__(self, num_classes=None, backbone="iresnet18", pretrained=False, mode="train"):
        super(Arcface, self).__init__()
        if backbone == "mobilefacenet":
            embedding_size = 128
            s = 32
            self.arcface = get_mbf(embedding_size=embedding_size, pretrained=pretrained)

        elif backbone == "mobilenetv1":
            embedding_size = 512
            s = 64
            self.arcface = get_mobilenet(dropout_keep_prob=0.5, embedding_size=embedding_size, pretrained=pretrained)

        elif backbone == "resnet":
            embedding_size = 512
            s = 64
            self.arcface = resnet18()

        elif backbone == "iresnet18":
            embedding_size = 512
            s = 64
            self.arcface = iresnet18(dropout_keep_prob=0.5, embedding_size=embedding_size)

        elif backbone == "iresnet34":
            embedding_size = 512
            s = 64
            self.arcface = iresnet34(dropout_keep_prob=0.5, embedding_size=embedding_size, pretrained=pretrained)

        elif backbone == "iresnet50":
            embedding_size = 512
            s = 64
            self.arcface = iresnet50(dropout_keep_prob=0.5, embedding_size=embedding_size, pretrained=pretrained)

        elif backbone == "iresnet100":
            embedding_size = 512
            s = 64
            self.arcface = iresnet100(dropout_keep_prob=0.5, embedding_size=embedding_size, pretrained=pretrained)

        elif backbone == "iresnet200":
            embedding_size = 512
            s = 64
            self.arcface = iresnet200()
        elif backbone == "resnet18":
            embedding_size = 512
            s = 64
            self.arcface = resnet18()
            self.header = nn.Sequential(
                nn.Linear(embedding_size, embedding_size),
                nn.ReLU(inplace=True),
                nn.Linear(embedding_size, 512)
            )
        elif backbone == "vit":
            embedding_size = 512
            s = 64
            self.arcface = VisionTransformer(embed_dim=512)


        elif backbone == "dilate":
            embedding_size = 512
            s = 64
            self.arcface = Dilateformer(depths=[2, 2], attn_depths=[6, 2], embed_dim=72, num_heads=[3, 6, 12, 24])

        elif backbone == "dilate2":
            embedding_size = 512
            s = 64
            self.arcface = Dilateformer2(depths=[2, 2, 2, 2], embed_dim=72, num_heads=[3, 6, 12, 24])

        elif backbone == "dilate3":
            embedding_size = 512
            s = 64
            self.arcface = Dilateformer3(depths=[2, 2, 2, 2], embed_dim=72, num_heads=[3, 6, 12, 24])

        elif backbone == "convformer":
            embedding_size = 512
            s = 64
            self.arcface = gm(modelname="SETR_ConvFormer", img_size=112, img_channel=3, classes=553)

        elif backbone == "vit1":
            embedding_size = 512
            s = 64
            self.arcface = get_model('vit_t', dropout=0.1, fp16=True)

        elif backbone == "resnet_vit":
            embedding_size = 512
            s = 64
            self.arcface = resnet_vit(num_classes=553, drop_rate=0.1)

        elif backbone == "fpn_vit":
            embedding_size = 512
            s = 64
            self.arcface = fpn_vit(num_classes=553, drop_rate=0.1)


        elif backbone == "resnet50":
            embedding_size = 512
            s = 64
            self.arcface = ResNet50(
                num_classes=553,resolution=(112,112)
            )

        elif backbone == "nextvit":
            embedding_size = 512
            s = 64
            self.arcface = nextvit_base(pretrained=False, pretrained_cfg=None)

        elif backbone == "swin_transformer":
            embedding_size = 512
            s = 64
            self.arcface = SwinTransformer(num_classes=553)


        elif backbone == "conformer":
            embedding_size = 512
            s = 64
            self.arcface = Conformer_small_patch16(pretrained=False)


        elif backbone == "vit_face":
            embedding_size = 512
            s = 64
            self.arcface = ViT(image_size=112,patch_size=16,num_classes=553,dim=512,depth=6,heads=6,mlp_dim=2048,pool='mean',dropout=0.1,emb_dropout=0.1)

        elif backbone == "t2tvit":
            embedding_size = 512
            s = 64
            self.arcface = T2TViT(image_size=112, num_classes=553, dim=512, depth=6, heads=16, mlp_dim=2048)

        elif backbone == "resnest":
            embedding_size = 512
            s = 64
            self.arcface = resnest(pretrained=False,num_classes=num_classes)

        elif backbone == "resnest_fpn":
            embedding_size = 512
            s = 64
            #self.arcface = ressnest_fpn(pretrained=False,num_classes)



        else:
            raise ValueError('Unsupported backbone - `{}`, Use mobilefacenet, mobilenetv1.'.format(backbone))
        self.mode = mode
        if mode == "train":
            self.head = Arcface_Head(embedding_size=embedding_size,num_classes=num_classes,s=s)
            #self.head = SubArcface_Head()

    def forward(self, x, y=None, mode="predict"):
        x = self.arcface(x)
        x = x.view(x.size()[0], -1)
        x = F.normalize(x)
        if mode == "predict":
            return x
        else:
            x = self.head(x, y)
            return x
