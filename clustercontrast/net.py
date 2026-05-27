import copy
import torch
import torch.nn as nn
from torch.nn import init
from torchvision import models
from torch.autograd import Variable
import torch.nn.functional as F
import math
from enum import Enum

from .resnet import resnet50


class ForwardMode(Enum):
    Train = 1
    TestRGB = 2
    TestIR = 3


def conv1x1(conv, x):
    x = x.unsqueeze(dim=-1).unsqueeze(dim=-1)
    x = conv(x)
    x = x.squeeze()
    return x


class Normalize(nn.Module):
    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        out = x.div(norm)
        return out


class ShallowLayer(nn.Module):
    """浅层"""
    def __init__(self, arch='resnet50'):
        super(ShallowLayer, self).__init__()

        model_v = resnet50(pretrained=True)
        self.shallow_layer = nn.Sequential(
            model_v.conv1,
            model_v.bn1,
            model_v.relu,
            model_v.maxpool
        )
        
        del model_v.layer1
        del model_v.layer2
        del model_v.layer3
        del model_v.layer4
        import gc
        gc.collect()

    def forward(self, x):
        x = self.shallow_layer(x)
        return x


class BaseModel(nn.Module):
    """深层，输入通道为64，第四层提取两次特征，提取动态信息"""
    def __init__(self, arch='resnet50') -> None:
        super().__init__()
        self.base = resnet50(pretrained=True, last_conv_stride=1, last_conv_dilation=1)
        self.layer4_temporal = copy.deepcopy(self.base.layer4)

    def forward(self, x):
        x = self.base.layer1(x)
        x = self.base.layer2(x)
        x = self.base.layer3(x)
        x_temporal = self.layer4_temporal(x)
        x = self.base.layer4(x)
        return x, x_temporal
    

class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )
        self.l = nn.Conv2d(channel, channel,1)
    def forward(self, x,f):
        y = self.fc(x)
        return f* y+f
    

class TemporalFeatLearning(nn.Module):
    def __init__(self,  ):
        super(TemporalFeatLearning, self).__init__()
        dim = 2048
        self.se_1 = SELayer(2048)
        self.se_2 = SELayer(2048)
        self.se_3 = SELayer(2048)
        self.se_4 = SELayer(2048)
        self.se_5 = SELayer(2048)
        self.se_6 = SELayer(2048)

        self.a = nn.Linear(dim, dim)
        self.b = nn.Linear(dim, dim)
        self.c = nn.Linear(dim, dim)
        self.d = nn.Linear(dim, dim)
        self.e = nn.Linear(dim, dim)
        self.f = nn.Linear(dim, dim)


    def forward(self, t_x,x,x_h):
        t1 = self.a(t_x)+x[0]
        t2 = self.b(t_x)+x[1]
        t3 = self.c(t_x)+x[2]
        t4 = self.d(t_x)+x[3]
        t5 = self.e(t_x)+x[4]
        t6 = self.f(t_x)+x[5]

        f1 = self.se_1(t1/2,x_h[0]).unsqueeze(dim=1)
        f2 = self.se_2(t2/2,x_h[1]).unsqueeze(dim=1)
        f3 = self.se_3(t3/2,x_h[2]).unsqueeze(dim=1)
        f4 = self.se_4(t4/2,x_h[3]).unsqueeze(dim=1)
        f5 = self.se_5(t5/2,x_h[4]).unsqueeze(dim=1)
        f6 = self.se_6(t6/2,x_h[5]).unsqueeze(dim=1)

        f = torch.cat((f1,f2,f3,f4,f5,f6),dim=1)
        f = f.mean(dim=1)

        return f
    

class Net(nn.Module):
    """
    视频数据经过
    模态独立浅层、共享深层、时序模块
    获得含有时序信息的模态相关的身份特征
    """
    def __init__(self, arch='resnet50') -> None:
        super().__init__()

        self.thermal_module = ShallowLayer(arch)
        self.visible_module = ShallowLayer(arch)
        self.base_model = BaseModel(arch)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.bottleneck = nn.BatchNorm1d(2048)

        self.lstm = nn.LSTM(2048, 2048, 2)
        self.temporal_module = TemporalFeatLearning()

        self.l2norm = Normalize(2)


    def forward(self, x, mode = ForwardMode.Train, seq_length=6):
        """
        x_v: 可见光图片数据
        x_t: 红外图片数据
        在训练模式下, x由相同长度的x_v, x_t拼接而成.
        在测试模式下, x是x_v或x_t
        """
        # 模态独立浅层
        if mode == ForwardMode.Train:
            x_v = x[:len(x)//2]
            x_t = x[len(x)//2:]
            x_v = self.visible_module(x_v)
            x_t = self.thermal_module(x_t)
            x = torch.cat((x_v, x_t), 0)
        elif mode == ForwardMode.TestRGB:
            x = self.visible_module(x)
            pass
        elif mode == ForwardMode.TestIR:
            x = self.thermal_module(x)
            pass
        else:
            raise ValueError(f'mode must be one of ForwardMode')

        # 模态共享层
        x, x_temporal = self.base_model(x)  # resnet50 输出的特征图

        # s = seq_length
        # x_h = self.avgpool(x).squeeze()
        # x_h = x_h.view(x_h.size(0)//s, s, -1).permute(1, 0, 2)         # (6, 32, 2048)
        # x_l = self.avgpool(x_temporal).squeeze()
        # x_l = x_l.view(x_temporal.size(0)//s, s, -1).permute(1, 0, 2)  # (6, 32, 2048)

        # h0 = torch.zeros(2, x_l.shape[1], x_l.shape[2]).cuda()
        # c0 = torch.zeros(2, x_l.shape[1], x_l.shape[2]).cuda()

        # if self.training: self.lstm.flatten_parameters()
        # output, (hn, cn) = self.lstm(x_l, (h0, c0))
        # x_pool = self.temporal_module(output[-1], x_l, x_h)

        # if self.training:
        #     return x_pool
        # else:
        #     feat = self.bottleneck(x_pool)
        #     return self.l2norm(feat)

        # without TRM
        x = self.avgpool(x).squeeze()
        
        

def create_net(arch):
    model = Net(arch)
    model.cuda()
    model = nn.DataParallel(model)
    return model