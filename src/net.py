
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from os.path import join
from torch.nn import Module, Sequential, Conv2d, ReLU,AdaptiveMaxPool2d, AdaptiveAvgPool2d, \
    NLLLoss, BCELoss, CrossEntropyLoss, AvgPool2d, MaxPool2d, Parameter, Linear, Sigmoid, Softmax, Dropout, Embedding



class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding=1):
        super().__init__()
        self.block = nn.Sequential(nn.Conv3d(in_channels, out_channels, kernel_size, padding=padding),
                                   nn.BatchNorm3d(out_channels),
                                   nn.ReLU(),
                                   nn.Conv3d(out_channels, out_channels, kernel_size, padding=padding),
                                   nn.BatchNorm3d(out_channels),
                                   nn.ReLU())

    def forward(self, x):
        out = self.block(x)
        return out


class Down(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size_pad, stride=2):
        super().__init__()
        self.block = nn.Sequential(nn.MaxPool3d(kernel_size_pad, stride=stride), DoubleConv(in_channels, out_channels, 3))

    def forward(self, x):
        out = self.block(x)
        return out


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size_up,padding=0,stride=2, out_pad=0, upsample=None):
        super().__init__()
        if upsample:
            self.up_s = nn.Upsample(scale_factor=2, mode=upsample, align_corners=True)
        else:
            self.up_s = nn.ConvTranspose3d(in_channels, in_channels // 2, kernel_size_up, stride=stride, padding=padding,
                                           output_padding=out_pad)

        self.convT = DoubleConv(in_channels, out_channels, 3)

    def forward(self, x1, x2):
        out = self.up_s(x1)
        out = self.convT(torch.cat((x2, out), dim=1))
        return out



class PAM_Module(nn.Module):
    """ Position attention module"""
    #Ref from SAGAN
    def __init__(self, in_dim):
        super(PAM_Module, self).__init__()
        self.chanel_in = in_dim

        self.query_conv = nn.Conv3d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.key_conv = nn.Conv3d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.value_conv = nn.Conv3d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = Parameter(torch.zeros(1))

        self.softmax = Softmax(dim=-1)
    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X (HxW) X (HxW)
        """
        m_batchsize, C, height, width, depth = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width*height*depth).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width*height*depth)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width*height*depth)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, height, width, depth)

        out = self.gamma*out + x
        return out


class CAM_Module(nn.Module):
    """ Channel attention module"""
    def __init__(self, in_dim):
        super(CAM_Module, self).__init__()
        self.chanel_in = in_dim


        self.gamma = Parameter(torch.zeros(1))
        self.softmax  = Softmax(dim=-1)
    def forward(self,x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X C X C
        """
        m_batchsize, C, height, width, depth = x.size()
        proj_query = x.view(m_batchsize, C, -1)
        proj_key = x.view(m_batchsize, C, -1).permute(0, 2, 1)
        energy = torch.bmm(proj_query, proj_key)
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy)-energy
        attention = self.softmax(energy_new)
        proj_value = x.view(m_batchsize, C, -1)

        out = torch.bmm(attention, proj_value)
        out = out.view(m_batchsize, C, height, width, depth)

        out = self.gamma*out + x
        return out


class DAB(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        inter_channels = in_channels // 4
        self.conv5a = nn.Sequential(nn.Conv3d(in_channels, inter_channels, 3, padding=1, bias=False),
                                   nn.BatchNorm3d(inter_channels),
                                   nn.ReLU())
        self.conv5c = nn.Sequential(nn.Conv3d(in_channels, inter_channels, 3, padding=1, bias=False),
                                   nn.BatchNorm3d(inter_channels),
                                   nn.ReLU())
        self.sa = PAM_Module(inter_channels)
        self.sc = CAM_Module(inter_channels)
        self.conv51 = nn.Sequential(nn.Conv3d(inter_channels, inter_channels, 3, padding=1, bias=False),
                                   nn.BatchNorm3d(inter_channels),
                                   nn.ReLU())
        self.conv52 = nn.Sequential(nn.Conv3d(inter_channels, inter_channels, 3, padding=1, bias=False),
                                   nn.BatchNorm3d(inter_channels),
                                   nn.ReLU())
        self.conv8 = nn.Sequential(nn.Dropout3d(0.1, False), nn.Conv3d(inter_channels, in_channels, 1))

    def forward(self, x):
        feat1 = self.conv5a(x)
        sa_feat = self.sa(feat1)
        sa_conv = self.conv51(sa_feat)

        feat2 = self.conv5c(x)
        sc_feat = self.sc(feat2)
        sc_conv = self.conv52(sc_feat)

        feat_sum = sa_conv + sc_conv
        sasc_output = self.conv8(feat_sum)

        return sasc_output



class Unet(nn.Module):
    def __init__(self, n_classes=1, upsample=False):
        super().__init__()
        self.n_classes = n_classes

        self.in1 = nn.Sequential(nn.Conv3d(14, 32, 3, padding=1), nn.BatchNorm3d(32), nn.ReLU())
        self.in2 = nn.Sequential(nn.Conv3d(32, 32, 3, padding=1), nn.BatchNorm3d(32), nn.ReLU())

        self.poll1_0 = nn.MaxPool3d(3, stride=2)
        self.down1_1 = nn.Sequential(nn.Conv3d(32, 64, 3, padding=1), nn.BatchNorm3d(64), nn.ReLU())
        self.down1_2 = nn.Sequential(nn.Conv3d(64, 64, 3, padding=1), nn.BatchNorm3d(64), nn.ReLU())

        self.poll2_0 = nn.MaxPool3d(3, stride=2)
        self.down2_1 = nn.Sequential(nn.Conv3d(64, 128, 3, padding=1), nn.BatchNorm3d(128), nn.ReLU())
        self.down2_2 = nn.Sequential(nn.Conv3d(128, 128, 3, padding=1), nn.BatchNorm3d(128), nn.ReLU())

        self.poll3_0 = nn.MaxPool3d(3, stride=2)
        self.down3_1 = nn.Sequential(nn.Conv3d(128, 256, 3, padding=1), nn.BatchNorm3d(256), nn.ReLU())
        self.down3_2 = nn.Sequential(nn.Conv3d(256, 256, 3, padding=1), nn.BatchNorm3d(256), nn.ReLU())

        factor = 2 if upsample else 1
        self.poll4_0 = nn.MaxPool3d(3, stride=2)
        self.down4_1 = nn.Sequential(nn.Conv3d(256, 512, 3, padding=1), nn.BatchNorm3d(512), nn.ReLU())
        self.down4_2 = nn.Sequential(nn.Conv3d(512, 512, 3, padding=1), nn.BatchNorm3d(512), nn.ReLU())

        self.upsample1_0 = nn.ConvTranspose3d(512, 512//2, 3, stride=2, padding=0, output_padding=0)
        self.up1_1 = nn.Sequential(nn.Conv3d(512, 256, 3, padding=1), nn.BatchNorm3d(256), nn.ReLU())
        self.up1_2 = nn.Sequential(nn.Conv3d(256, 256, 3, padding=1), nn.BatchNorm3d(256), nn.ReLU())

        self.upsample2_0 = nn.ConvTranspose3d(256, 256 // 2, 3, stride=2, padding=0, output_padding=0)
        self.up2_1 = nn.Sequential(nn.Conv3d(256, 128, 3, padding=1), nn.BatchNorm3d(128), nn.ReLU())
        self.up2_2 = nn.Sequential(nn.Conv3d(128, 128, 3, padding=1), nn.BatchNorm3d(128), nn.ReLU())

        self.upsample3_0 = nn.ConvTranspose3d(128, 128 // 2, 3, stride=2, padding=0, output_padding=1)
        self.up3_1 = nn.Sequential(nn.Conv3d(128, 64, 3, padding=1), nn.BatchNorm3d(64), nn.ReLU())
        self.up3_2 = nn.Sequential(nn.Conv3d(64, 64, 3, padding=1), nn.BatchNorm3d(64), nn.ReLU())

        self.upsample4_0 = nn.ConvTranspose3d(64, 64 // 2, 3, stride=2, padding=0, output_padding=0)
        self.up4_1 = nn.Sequential(nn.Conv3d(64, 32, 3, padding=1), nn.BatchNorm3d(32), nn.ReLU())
        self.up4_2 = nn.Sequential(nn.Conv3d(32, 32, 3, padding=1), nn.BatchNorm3d(32), nn.ReLU())

        self.conv = nn.Conv3d(32, self.n_classes, 1)
        self.sigmoid = nn.Sigmoid()

        self.up1 = nn.ConvTranspose3d(512, 512//2, 3, stride=2, padding=0, output_padding=0)
        self.up2 = nn.ConvTranspose3d(256, 256 // 2, 3, stride=2, padding=0, output_padding=0)
        self.up3 = nn.ConvTranspose3d(128, 128 // 2, 3, stride=2, padding=0, output_padding=1)
        self.up4 = nn.ConvTranspose3d(64, 64 // 2, 3, stride=2, padding=0, output_padding=0)

        # self.dab0 = DAB(32)
        # self.dab1 = DAB(64)
        self.dab2 = DAB(128)
        self.dab3 = DAB(256)
        self.dab4 = DAB(512)



    def refine(self, last_layer, current_layer, up_layer):
        up_sampled = up_layer(last_layer)
        mask = torch.where((0 < up_sampled) & (up_sampled < 1), 1, 0)
        refined = current_layer*mask + up_sampled * (1 - mask)
        return refined


    def forward(self, x):
        x1 = self.in1(x)
        # x1 = self.dab0(x1)
        x1 = self.in2(x1)
        # print(x1.shape)

        x1_0 = self.poll1_0(x1)
        x1_1 = self.down1_1(x1_0)
        # x1_1 = self.dab1(x1_1)
        x2 = self.down1_2(x1_1)
        # print(x2.shape)

        x2_0 = self.poll2_0(x2)
        x2_1 = self.down2_1(x2_0)
        x2_1 = self.dab2(x2_1)
        x3 = self.down2_2(x2_1)
        # print(x3.shape)

        x3_0 = self.poll3_0(x3)
        x3_1 = self.down3_1(x3_0)
        x3_1 = self.dab3(x3_1)
        x4 = self.down3_2(x3_1)
        # print(x4.shape)

        x4_0 = self.poll4_0(x4)
        x4_1 = self.down4_1(x4_0)
        x4_1 = self.dab4(x4_1)
        x5 = self.down4_2(x4_1)
        # print(x5.shape)
        last_layer_1 = x5

        # print('=============================')
        x11_0 = self.upsample1_0(x5)
        x11_1 = self.up1_1(torch.cat((x11_0, x4), dim=1))
        x11 = self.up1_2(x11_1)
        # print(x11.shape)
        last_layer_2 = self.refine(last_layer_1, x11, self.up1)

        x22_0 = self.upsample2_0(x11)
        x22_1 = self.up2_1(torch.cat((x22_0, x3), dim=1))
        x22 = self.up2_2(x22_1)
        # print(x22.shape)
        last_layer_3 = self.refine(last_layer_2, x22, self.up2)

        x33_0 = self.upsample3_0(x22)
        x33_1 = self.up3_1(torch.cat((x33_0, x2), dim=1))
        x33 = self.up3_2(x33_1)
        # print(x33.shape)
        last_layer_4 = self.refine(last_layer_3, x33, self.up3)

        x44_0 = self.upsample4_0(x33)
        x44_1 = self.up4_1(torch.cat((x44_0, x1), dim=1))
        x44 = self.up4_2(x44_1)
        # print(x44.shape)
        last_layer_5 = self.refine(last_layer_4, x44, self.up4)

        # logits = self.conv(x44)
        logits = self.conv(last_layer_5)
        prob = self.sigmoid(logits)
        return prob




if __name__ == '__main__':
    model = Unet()
    input = torch.randn(size=(1, 14, 65, 65, 65))
    output = model(input)
    print('output,shape=', output.shape)


