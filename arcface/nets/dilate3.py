import torch
import torch.nn as nn
from functools import partial
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg
import torch.nn as nn
import torch.nn.functional as F
import math


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_planes, planes, 1, bias=False),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True),
            nn.Conv2d(planes, planes, 3, stride, 1, bias=False),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True),
            nn.Conv2d(planes, self.expansion * planes, 1, bias=False),
            nn.BatchNorm2d(self.expansion * planes),
        )
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.bottleneck(x)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out


class FPN(nn.Module):
    def __init__(self, layers):
        super(FPN, self).__init__()
        self.inplanes = 72
        self.conv1 = nn.Conv2d(3, 72, 7, 2, 3, bias=False)
        self.bn1 = nn.BatchNorm2d(72)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(3, 2, 1)

        self.layer1 = self._make_layer(36, layers[0], 2)
        self.layer2 = self._make_layer(72, layers[1], 2)
        self.layer3 = self._make_layer(144, layers[2], 2)
        self.layer4 = self._make_layer(288, layers[3], 2)
        self.toplayer = nn.Conv2d(2304, 72, 1, 1, 0)

        self.smooth1 = nn.Conv2d(576, 576, 3, 1, 1)
        self.smooth2 = nn.Conv2d(576, 576, 3, 1, 1)
        self.smooth3 = nn.Conv2d(72, 72, 3, 1, 1)

        self.latlayer1 = nn.Conv2d(1152, 576, 1, 1, 0)
        self.latlayer2 = nn.Conv2d(576, 576, 1, 1, 0)
        self.latlayer3 = nn.Conv2d(288, 576, 1, 1, 0)

    def _make_layer(self, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != Bottleneck.expansion * planes:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, Bottleneck.expansion * planes, 1, stride, bias=False),
                nn.BatchNorm2d(Bottleneck.expansion * planes)
            )
        layers = []
        layers.append(Bottleneck(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * Bottleneck.expansion
        for i in range(1, blocks):
            layers.append(Bottleneck(self.inplanes, planes))
        return nn.Sequential(*layers)

    def _upsample_add(self, x, y):
        _, _, H, W = y.shape
        return F.upsample(x, size=(H, W), mode='bilinear') + y

    def forward(self, x):

        c1 = self.relu(self.bn1(self.conv1(x)))
        c2 = self.layer1(c1)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)

        # p5 = self.toplayer(c5)
        # p4 = self._upsample_add(p5, self.latlayer1(c4))
        # p3 = self._upsample_add(p4, self.latlayer2(c3))
        # p2 = self._upsample_add(p3, self.latlayer3(c2))

        # p4 = self.smooth1(p4)
        # p2 = self.smooth3(p2)
        return c1, c2, c3, c4





class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class DilateAttention(nn.Module):
    "Implementation of Dilate-attention"

    def __init__(self, head_dim, qk_scale=None, attn_drop=0, kernel_size=3, dilation=1):
        super().__init__()
        self.head_dim = head_dim
        self.scale = qk_scale or head_dim ** -0.5
        self.kernel_size = kernel_size
        self.unfold = nn.Unfold(kernel_size, dilation, dilation * (kernel_size - 1) // 2, 1)
        self.attn_drop = nn.Dropout(attn_drop)

    def forward(self, q, k, v):
        # B, C//3, H, W
        B, d, H, W = q.shape
        q = q.reshape([B, d // self.head_dim, self.head_dim, 1, H * W]).permute(0, 1, 4, 3, 2)  # B,h,N,1,d
        k = self.unfold(k).reshape(
            [B, d // self.head_dim, self.head_dim, self.kernel_size * self.kernel_size, H * W]).permute(0, 1, 4, 2,
                                                                                                        3)  # B,h,N,d,k*k
        attn = (q @ k) * self.scale  # B,h,N,1,k*k
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        v = self.unfold(v).reshape(
            [B, d // self.head_dim, self.head_dim, self.kernel_size * self.kernel_size, H * W]).permute(0, 1, 4, 3,
                                                                                                        2)  # B,h,N,k*k,d
        x = (attn @ v).transpose(1, 2).reshape(B, H, W, d)
        return x


class MultiDilatelocalAttention(nn.Module):
    "Implementation of Dilate-attention"

    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None,
                 attn_drop=0., proj_drop=0., kernel_size=3, dilation=[1, 2, 3]):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.dilation = dilation
        self.kernel_size = kernel_size
        self.scale = qk_scale or head_dim ** -0.5
        self.num_dilation = len(dilation)
        assert num_heads % self.num_dilation == 0, f"num_heads{num_heads} must be the times of num_dilation{self.num_dilation}!!"
        self.qkv = nn.Conv2d(dim, dim * 3, 1, bias=qkv_bias)
        self.dilate_attention = nn.ModuleList(
            [DilateAttention(head_dim, qk_scale, attn_drop, kernel_size, dilation[i])
             for i in range(self.num_dilation)])
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, H, W, C = x.shape
        x = x.permute(0, 3, 1, 2)  # B, C, H, W
        qkv = self.qkv(x).reshape(B, 3, self.num_dilation, C // self.num_dilation, H, W).permute(2, 1, 0, 3, 4, 5)
        # num_dilation,3,B,C//num_dilation,H,W
        x = x.reshape(B, self.num_dilation, C // self.num_dilation, H, W).permute(1, 0, 3, 4, 2)
        # num_dilation, B, H, W, C//num_dilation
        for i in range(self.num_dilation):
            x[i] = self.dilate_attention[i](qkv[i][0], qkv[i][1], qkv[i][2])  # B, H, W,C//num_dilation
        x = x.permute(1, 2, 3, 0, 4).reshape(B, H, W, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class DilateBlock(nn.Module):
    "Implementation of Dilate-attention block"

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, kernel_size=3, dilation=[1, 2, 3],
                 cpe_per_block=False):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.cpe_per_block = cpe_per_block
        if self.cpe_per_block:
            self.pos_embed = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.norm1 = norm_layer(dim)
        self.attn = MultiDilatelocalAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                                              attn_drop=attn_drop, kernel_size=kernel_size, dilation=dilation)

        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim,
                       act_layer=act_layer, drop=drop)

    def forward(self, x):
        if self.cpe_per_block:
            x = x + self.pos_embed(x)
        x = x.permute(0, 2, 3, 1)
        x = self.norm1(x)
        x = x + self.drop_path(self.attn(x))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        x = x.permute(0, 3, 1, 2)
        # B, C, H, W
        return x


class GlobalAttention(nn.Module):
    "Implementation of self-attention"

    def __init__(self, dim, num_heads=8, qkv_bias=False,
                 qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        #x = x.permute(0, 3, 2, 1)  # B, H, W, C
        B, H, W, C = x.shape
        qkv = self.qkv(x).reshape(B, H * W, 3, self.num_heads,
                                  C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, H, W, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class GlobalBlock(nn.Module):
    """
    Implementation of Transformer
    """

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0.,
                 attn_drop=0., drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 cpe_per_block=False):
        super().__init__()
        self.cpe_per_block = cpe_per_block
        if self.cpe_per_block:
            self.pos_embed = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.norm1 = norm_layer(dim)
        self.attn = GlobalAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias,
                                    qk_scale=qk_scale, attn_drop=attn_drop)

        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim,
                       act_layer=act_layer, drop=drop)

    def forward(self, x):
        if self.cpe_per_block:
            x = x + self.pos_embed(x)
        x = x.permute(0, 2, 3, 1)
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        x = x.permute(0, 3, 1, 2)
        return x





class PatchMerging(nn.Module):
    """ Patch Merging Layer.
    """

    def __init__(self, in_channels, out_channels, merging_way, cpe_per_satge, norm_layer=nn.BatchNorm2d):
        super().__init__()
        assert merging_way in ['conv3_2', 'conv2_2', 'avgpool3_2', 'avgpool2_2'], \
            "the merging way is not exist!"
        self.cpe_per_satge = cpe_per_satge
        if merging_way == 'conv3_2':
            self.proj = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
                norm_layer(out_channels),
            )
        elif merging_way == 'conv2_2':
            self.proj = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=2, stride=2, padding=0),
                norm_layer(out_channels),
            )
        elif merging_way == 'avgpool3_2':
            self.proj = nn.Sequential(
                nn.AvgPool2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
                norm_layer(out_channels),
            )
        else:
            self.proj = nn.Sequential(
                nn.AvgPool2d(in_channels, out_channels, kernel_size=2, stride=2, padding=0),
                norm_layer(out_channels),
            )
        if self.cpe_per_satge:
            self.pos_embed = nn.Conv2d(out_channels, out_channels, 3, padding=1, groups=out_channels)

    def forward(self, x):
        # x: B, C, H ,W
        x = self.proj(x)
        if self.cpe_per_satge:
            x = x + self.pos_embed(x)
        return x


class Dilatestage(nn.Module):
    """ A basic Dilate Transformer layer for one stage.
    """

    def __init__(self, dim, depth, num_heads, kernel_size, dilation,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0.,
                 attn_drop=0., drop_path=0., act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm, cpe_per_satge=False, cpe_per_block=False,
                 merging_way=None):
        super().__init__()
        # build blocks
        self.blocks = nn.ModuleList([
            DilateBlock(dim=dim, num_heads=num_heads,
                        kernel_size=kernel_size, dilation=dilation,
                        mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                        qk_scale=qk_scale, drop=drop, attn_drop=attn_drop,
                        drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                        norm_layer=norm_layer, act_layer=act_layer, cpe_per_block=cpe_per_block)
            for i in range(depth)])
        # patch merging layer

    def forward(self, x):
        a = 0
        for blk in self.blocks:
            x = blk(x)
        return x


class Globalstage(nn.Module):
    """ A basic Transformer layer for one stage."""

    def __init__(self, dim, depth, num_heads, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop=0., attn_drop=0., drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 cpe_per_satge=False, cpe_per_block=False,
                 downsample=True, merging_way=None):
        super().__init__()
        # build blocks
        self.blocks = nn.ModuleList([
            GlobalBlock(dim=dim, num_heads=num_heads,
                        mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                        qk_scale=qk_scale, drop=drop, attn_drop=attn_drop,
                        drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                        norm_layer=norm_layer, act_layer=act_layer, cpe_per_block=cpe_per_block)
            for i in range(depth)])

        # patch merging layer

    def forward(self, x):
        a = 0
        for blk in self.blocks:
            x = blk(x)
        return x


class Dilateformer3(nn.Module):
    def __init__(self, img_size=112, patch_size=4, in_chans=3, num_classes=1000, embed_dim=256,
                 depths=[2, 2, 2, 2], num_heads=[3, 6, 12, 24], kernel_size=3, dilation=[1, 2, 3],
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.1,
                 norm_layer=partial(nn.LayerNorm, eps=1e-6),
                 merging_way='conv3_2',
                 patch_way='overlaping',
                 dilate_attention=[True, True],
                 attention=[False,False],
                 cpe_per_satge=False, cpe_per_block=True):
        super().__init__()
        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.layers = len(depths)
        self.embed_dim = embed_dim
        self.num_features = int(embed_dim * 2 ** (self.layers - 1))
        self.mlp_ratio = mlp_ratio
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        self.fpn = FPN(layers=[2, 2, 2, 2])
        self.conv1 = nn.Conv2d(3, 72, 7, 2, 3, bias=False)
        self.bn1 = nn.BatchNorm2d(72)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(3, 2, 1)
        # patch embedding
        dpr = [x.item() for x in torch.linspace(0, drop_path, sum(depths))]
        self.Dilatestage1 = Dilatestage(dim=int(embed_dim * 2 ** 0),
                                    depth=depths[0],
                                    num_heads=num_heads[0],
                                    kernel_size=kernel_size,
                                    dilation=dilation,
                                    mlp_ratio=self.mlp_ratio,
                                    qkv_bias=qkv_bias, qk_scale=qk_scale,
                                    drop=drop, attn_drop=attn_drop,
                                    drop_path=dpr[sum(depths[:0]):sum(depths[:0 + 1])],
                                    norm_layer=norm_layer,
                                    cpe_per_block=cpe_per_block,
                                    cpe_per_satge=cpe_per_satge,
                                    merging_way=merging_way
                                    )
        self.Dilatestage2 = Dilatestage(dim=int(embed_dim * 2 ** 1),
                                        depth=depths[1],
                                        num_heads=num_heads[1],
                                        kernel_size=kernel_size,
                                        dilation=dilation,
                                        mlp_ratio=self.mlp_ratio,
                                        qkv_bias=qkv_bias, qk_scale=qk_scale,
                                        drop=drop, attn_drop=attn_drop,
                                        drop_path=dpr[sum(depths[:1]):sum(depths[:1 + 1])],
                                        norm_layer=norm_layer,
                                        cpe_per_block=cpe_per_block,
                                        cpe_per_satge=cpe_per_satge,
                                        merging_way=merging_way
                                        )
        self.Globalstage3 = Globalstage(dim=int(embed_dim * 2 ** 2),
                                    depth=depths[2],
                                    num_heads=num_heads[2],
                                    mlp_ratio=self.mlp_ratio,
                                    qkv_bias=qkv_bias, qk_scale=qk_scale,
                                    drop=drop, attn_drop=attn_drop,
                                    drop_path=dpr[sum(depths[:2]):sum(depths[:2 + 1])],
                                    norm_layer=norm_layer,
                                    cpe_per_block=cpe_per_block,
                                    cpe_per_satge=cpe_per_satge,
                                    merging_way=merging_way
                                    )
        self.Globalstage4 = Globalstage(dim=int(embed_dim * 2 ** 3),
                                        depth=depths[3],
                                        num_heads=num_heads[3],
                                        mlp_ratio=self.mlp_ratio,
                                        qkv_bias=qkv_bias, qk_scale=qk_scale,
                                        drop=drop, attn_drop=attn_drop,
                                        drop_path=dpr[sum(depths[:3]):sum(depths[:3 + 1])],
                                        norm_layer=norm_layer,
                                        cpe_per_block=cpe_per_block,
                                        cpe_per_satge=cpe_per_satge,
                                        merging_way=merging_way
                                        )

        self.norm = norm_layer(self.num_features)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(p=0.1, inplace=True)
        self.head = nn.Sequential(
            nn.Linear(self.num_features, 512),
            nn.BatchNorm1d(512)
        )
        self.downsample1 = PatchMerging(72, 576, merging_way, cpe_per_satge)
        self.downsample2 = PatchMerging(144, 576, merging_way, cpe_per_satge)
        self.downsample3 = PatchMerging(288, 576, merging_way, cpe_per_satge)
        self.apply(self._init_weights)
        self.latlayer0 = nn.Conv2d(1152, 576, 1, 1, 0)
        self.latlayer1 = nn.Conv2d(576, 576, 1, 1, 0)
        self.latlayer2 = nn.Conv2d(288, 576, 1, 1, 0)
        self.latlayer3 = nn.Conv2d(144, 576, 1, 1, 0)
        self.bn2 = nn.BatchNorm1d(576)

    def _upsample_add(self, x, y):
        _, _, H, W = y.shape
        return F.upsample(x, size=(H, W), mode='bilinear') + y


    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    def forward_features(self, x):
        # x = self.patch_embed(x)
        # x = self.dcn(x)
        c1, c2, c3, c4 = self.fpn(x)
        c1 = self.Dilatestage1(c1)
        c2 = self.Dilatestage2(c2)
        c3 = self.Globalstage3(c3)
        c4 = self.Globalstage4(c4)

        p1 = self.avgpool(self.norm(self.downsample1(c1).flatten(2).transpose(1, 2)).transpose(1, 2))
        p2 = self.avgpool(self.norm(self.downsample2(c2).flatten(2).transpose(1, 2)).transpose(1, 2))
        p3 = self.avgpool(self.norm(self.downsample3(c3).flatten(2).transpose(1, 2)).transpose(1, 2))
        p4 = self.avgpool(self.norm(c4.flatten(2).transpose(1, 2)).transpose(1, 2))
        #p = p1+p2+p3+p4
        p1 = self.bn2(p1)
        p2 = self.bn2(p2)
        p3 = self.bn2(p3)
        p4 = self.bn2(p4)
        p = p1+p2+p3+p4
        #p = torch.cat((p1,p2,p3,p4),1)
        x = torch.flatten(p, 1)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.dropout(x)
        x = self.head(x)
        return x





