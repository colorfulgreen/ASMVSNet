import torch
import torch.nn as nn
import torch.nn.functional as F

from net.layers.deform_layers import DeformConv2d


class ConvBnReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, pad=1, dilation=1, mdconv=False):
        super(ConvBnReLU, self).__init__()
        if mdconv:
            self.conv = DeformConv2d(in_channels, out_channels, kernel_size, stride=stride, dilation=dilation, bias=False)
        else:
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=pad, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self,x):
        return F.relu(self.bn(self.conv(x)), inplace=True)


class ConvReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, pad=1, dilation=1, mdconv=False):
        super(ConvReLU, self).__init__()
        if mdconv:
            self.conv = DeformConv2d(in_channels, out_channels, kernel_size, stride=stride, dilation=dilation, bias=False)
        else:
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=pad, dilation=dilation, bias=False)

    def forward(self,x):
        return F.relu(self.conv(x), inplace=True)


class ConvBn(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, pad=1, dilation=1, mdconv=False):
        super(ConvBn, self).__init__()
        if mdconv:
            self.conv = DeformConv2d(in_channels, out_channels, kernel_size, stride=stride, dilation=dilation, bias=False)
        else:
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=pad, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self,x):
        return self.bn(self.conv(x))


class ResidualBlock(nn.Module):
    def __init__(self, in_planes, planes, stride=1, mdconv=False):
        super(ResidualBlock, self).__init__()
        self.conv1 = ConvBnReLU(in_planes, planes, 3, stride=stride, pad=1, mdconv=mdconv)
        self.conv2 = ConvBn(planes, planes, 3, stride=1, pad=1, mdconv=mdconv)

        self.relu = nn.ReLU(inplace=True)

        if stride == 1:
            self.downsample = None
        else:    
            self.downsample = ConvBn(in_planes, planes, 3, stride=stride, pad=1)

    def forward(self, x):
        y = self.conv2(self.conv1(x))
        if self.downsample is not None:
            x = self.downsample(x)
        return self.relu(x+y)


class ContextNet(nn.Module):
    '''3d position augmented FPN'''
    def __init__(self, test=False, mdconv=True, base_chs=None, use_dynamic_feature=False):
        super(ContextNet, self).__init__()
        self.in_planes = 8
        self.test = test
        self.use_dynamic_feature = use_dynamic_feature
        base_chs = base_chs or [48, 32, 16]

        self.conv1 = ConvBnReLU(3,8)
        self.layer1 = self._make_layer(16, stride=2, last_mdconv=mdconv)
        self.layer2 = self._make_layer(32, stride=2)
        self.layer3 = self._make_layer(48, stride=2, mdconv=mdconv)

        # output convolution
        self.output = nn.Conv2d(48, 16, 3, stride=1, padding=1)

        self.inner2 = nn.Conv2d(32, 48, 1, stride=1, padding=0, bias=True)
        self.inner1 = nn.Conv2d(16, 48, 1, stride=1, padding=0, bias=True)

    def _make_layer(self, dim, stride=1, mdconv=False, last_mdconv=False):   
        layer1 = ResidualBlock(self.in_planes, dim, stride=stride, mdconv=mdconv)
        layer2 = ResidualBlock(dim, dim, mdconv=mdconv or last_mdconv)
        layers = (layer1, layer2)
        
        self.in_planes = dim
        return nn.Sequential(*layers)

    def forward(self, x, proj_invs_l3, context_transform, is_training=False):
        B,_,H,W = x.size()
        fea0 = self.conv1(x)
        fea1 = self.layer1(fea0)
        fea2 = self.layer2(fea1)
        fea3 = self.layer3(fea2)

        if self.use_dynamic_feature:
            aug_fea3, aug_feas = context_transform(fea3, proj_invs_l3)
        else:
            aug_fea3 = fea3

        intra_feat = F.interpolate(aug_fea3, scale_factor=2, mode="nearest") + self.inner2(fea2)
        intra_feat = F.interpolate(intra_feat, scale_factor=2, mode="nearest") + self.inner1(fea1)
        fea_level1 = self.output(intra_feat)
        return fea_level1


class FeatureNet(nn.Module):
    '''Returns: [16xH/4xW/4, 8xH/2xW/2]'''
    def __init__(self, test=False, mdconv=True, base_chs=None, use_dynamic_feature=False):
        super(FeatureNet, self).__init__()
        self.in_planes = 8
        self.test = test
        self.use_dynamic_feature = use_dynamic_feature
        base_chs = base_chs or [48, 32, 16]

        self.conv1 = ConvBnReLU(3,8)
        self.layer1 = self._make_layer(16, stride=2, last_mdconv=mdconv)
        self.layer2 = self._make_layer(32, stride=2)
        self.layer3 = self._make_layer(48, stride=2, mdconv=mdconv)

        # output convolution
        self.output2 = nn.Conv2d(48, 16, 3, stride=1, padding=1)
        self.output1 = nn.Conv2d(48, 8, 3, stride=1, padding=1)

        self.inner1 = nn.Conv2d(16, 48, 1, stride=1, padding=0, bias=True)
        self.inner2 = nn.Conv2d(32, 48, 1, stride=1, padding=0, bias=True)

    def _make_layer(self, dim, stride=1, mdconv=False, last_mdconv=False):   
        layer1 = ResidualBlock(self.in_planes, dim, stride=stride, mdconv=mdconv)
        layer2 = ResidualBlock(dim, dim, mdconv=mdconv or last_mdconv)
        layers = (layer1, layer2)
        
        self.in_planes = dim
        return nn.Sequential(*layers)

    def forward(self, x, proj_invs_l3, feature_transform, feature_transform_infer, is_training=False):
        feas={"level2":[],"level1":[]}

        B,V,_,H,W = x.size()

        x = x.view(B*V, -1, H, W)
        fea0 = self.conv1(x)
        fea1 = self.layer1(fea0)
        fea2 = self.layer2(fea1)
        fea3 = self.layer3(fea2)

        if is_training:
            fea3 = torch.unbind(fea3.view(B,V,-1,H//8,W//8), dim=1)
            aug_fea3_list = []
            proj_invs_l3 = torch.unbind(proj_invs_l3, dim=1) 
            ref_aug_feas = None 
            for i in range(V):
                if self.use_dynamic_feature:
                    aug_fea, aug_feas = feature_transform(fea3[i], proj_invs_l3[i], 
                                                          ref_aug_feas=ref_aug_feas, is_ref=i==0)
                    if i == 0:
                        ref_aug_feas = aug_feas 
                else:
                    aug_fea = fea3[i]
                aug_fea3_list.append(aug_fea)
            aug_fea3 = torch.stack(aug_fea3_list, dim=1).reshape(B*V, -1, H//8, W//8) 
        else:
            if self.use_dynamic_feature:
                proj_invs_l3 = proj_invs_l3.view(B*V, 3, 3)
                aug_fea3 = feature_transform_infer(fea3, proj_invs_l3)
            else:
                aug_fea3 = fea3

        aug_fea3 = aug_fea3.half()
        intra_feat = F.interpolate(aug_fea3, scale_factor=2, mode="nearest") + self.inner2(fea2)
        feas["level2"] = torch.unbind(self.output2(intra_feat).view(B,V,-1,H//4,W//4), dim=1)

        intra_feat = F.interpolate(intra_feat, scale_factor=2, mode="nearest") + self.inner1(fea1)
        feas["level1"] = torch.unbind(self.output1(intra_feat).view(B,V,-1,H//2,W//2), dim=1)

        return feas
