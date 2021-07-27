import torch
import torch.nn as nn
import numpy as np
import math
from torchsummary import summary
from math import sqrt
import matplotlib.pyplot as plt
from dcn.modules.deform_conv import *
import functools
import torch.nn.functional as F


class DNL_3DCNN(nn.Module):
    def __init__(self, upscale_factor, in_channel=1, out_channel=1, n_feats=64):
        super(DNL_3DCNN, self).__init__()
        self.upscale_factor = upscale_factor
        self.in_channel = in_channel

        self.input = nn.Sequential(
            nn.Conv3d(in_channels=in_channel, out_channels=n_feats, kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(negative_slope=0.1, inplace=True)
        )

        self.residual_layer = self.make_layer(functools.partial(ResBlock_3d, n_feats), 5)
        self.TA = nn.Conv2d(7 * n_feats, n_feats, 1, 1, bias=True)
        # Non-local Attention Module #
        self.non_local = NonLocalBlock(n_feats, n_feats)
        ### reconstruct
        #self.reconstruct = self.make_layer(functools.partial(ResBlock, n_feats), 6)
        self.reconstruct = RRDBNet(n_feats ,n_feats , nb=23, gc=32)
        ###upscale
        self.upscale = nn.Sequential(
            nn.Conv2d(n_feats, n_feats * upscale_factor ** 2, 1, 1, 0, bias=False),
            nn.PixelShuffle(upscale_factor),
            nn.Conv2d(n_feats, out_channel, 3, 1, 1, bias=False),
            nn.Conv2d(out_channel, out_channel, 3, 1, 1, bias=False)
        )

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
                m.weight.data.normal_(0, sqrt(2. / n))

    def make_layer(self, block, num_of_layer):
        layers = []
        for _ in range(num_of_layer):
            layers.append(block())
        return nn.Sequential(*layers)
    def forward(self, x):
        b, c, n, h, w = x.size()
        residual = F.interpolate(x[:, :, n // 2, :, :], scale_factor=self.upscale_factor, mode='bilinear',
                                 align_corners=False)
        out = self.input(x)
        out = self.residual_layer(out)
        out = self.non_local(out)
        out = self.TA(out.permute(0,2,1,3,4).contiguous().view(b, -1, h, w))  # B, C, H, W
        out = self.reconstruct(out)
        ###upscale
        out = self.upscale(out)
        out = torch.add(out, residual)
        return out

class RRDBNet(nn.Module):
    # def __init__(self, in_nc, out_nc, n_feats, nb, gc=32):
    def __init__(self, in_nc, n_feats, nb, gc=32):
        super(RRDBNet, self).__init__()
        RRDB_block_f = functools.partial(RRDB, n_feats=n_feats, gc=gc)
        self.conv_first = nn.Conv2d(in_nc, n_feats, 3, 1, 1, bias=True)
        self.RRDB_trunk = make_layer(RRDB_block_f, nb)
        self.trunk_conv = nn.Conv2d(n_feats, n_feats, 3, 1, 1, bias=True)

    def forward(self, x):
        fea = self.conv_first(x)
        trunk = self.trunk_conv(self.RRDB_trunk(fea))
        return trunk

class NonLocalBlock(nn.Module):
    def __init__(self, in_channels, inter_channels):
        super(NonLocalBlock, self).__init__()

        self.in_channels = in_channels
        self.inter_channels = inter_channels

        self.g = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1, stride=1,
                           padding=0)

        self.W = nn.Conv2d(in_channels=self.inter_channels, out_channels=self.in_channels, kernel_size=1, stride=1,
                           padding=0)
        nn.init.constant_(self.W.weight, 0)
        nn.init.constant_(self.W.bias, 0)

        self.theta = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1,
                               stride=1,
                               padding=0)

        self.phi = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1,
                             stride=1,
                             padding=0)

        self.act = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, supp_feature, ref_feature):
        x = supp_feature  # b,c,h,w
        y = ref_feature

        batch_size = x.size(0)

        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)

        theta_x = theta_x.permute(0, 2, 1)

        phi_y = self.phi(y).view(batch_size, self.inter_channels, -1)

        g_y = self.g(y).view(batch_size, self.inter_channels, -1)

        g_y = g_y.permute(0, 2, 1)

        f = torch.matmul(theta_x, phi_y)

        f_div_C = F.softmax(f, dim=1)

        x1 = torch.matmul(f_div_C, g_y)

        x1 = x1.permute(0, 2, 1).contiguous()

        x1 = x1.view(batch_size, self.inter_channels, *supp_feature.size()[2:])
        W_x1 = self.W(x1)
        z = x + W_x1

        return z



class ResBlock_3d(nn.Module):
    def __init__(self, n_feats):
        super(ResBlock_3d, self).__init__()
        self.dcn0 = DeformConvPack_d(n_feats, n_feats, kernel_size=3, stride=1, padding=1, dimension='HW')
        self.dcn1 = DeformConvPack_d(n_feats, n_feats, kernel_size=3, stride=1, padding=1, dimension='HW')
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, x):
        return self.dcn1(self.lrelu(self.dcn0(x))) + x

class ResBlock(nn.Module):
    def __init__(self, n_feats):
        super(ResBlock, self).__init__()
        self.dcn0 = nn.Conv2d(n_feats, n_feats, kernel_size=3, stride=1, padding=1)
        self.dcn1 = nn.Conv2d(n_feats, n_feats, kernel_size=3, stride=1, padding=1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, x):
        return self.dcn1(self.lrelu(self.dcn0(x))) + x

class RRDB(nn.Module):
    '''Residual in Residual Dense Block'''

    def __init__(self, n_feats, gc=32):
        super(RRDB, self).__init__()
        self.RDB1 = ResidualDenseBlock_5C(n_feats, gc)
        self.RDB2 = ResidualDenseBlock_5C(n_feats, gc)
        self.RDB3 = ResidualDenseBlock_5C(n_feats, gc)

    def forward(self, x):
        out = self.RDB1(x)
        out = self.RDB2(out)
        out = self.RDB3(out)
        return out * 0.2 + x

def make_layer(block, n_layers):
    layers = []
    for _ in range(n_layers):
        layers.append(block())
    return nn.Sequential(*layers)

class ResidualDenseBlock_5C(nn.Module):
    def __init__(self, n_feats=64, gc=32, bias=True):
        super(ResidualDenseBlock_5C, self).__init__()
        # gc: growth channel, i.e. intermediate channels
        self.conv1 = nn.Conv2d(n_feats, gc, 3, 1, 1, bias=bias)
        self.conv2 = nn.Conv2d(n_feats + gc, gc, 3, 1, 1, bias=bias)
        self.conv3 = nn.Conv2d(n_feats + 2 * gc, gc, 3, 1, 1, bias=bias)
        self.conv4 = nn.Conv2d(n_feats + 3 * gc, gc, 3, 1, 1, bias=bias)
        self.conv5 = nn.Conv2d(n_feats + 4 * gc, n_feats, 3, 1, 1, bias=bias)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x

if __name__ == "__main__":
    #net = DNL_3DCNN(4).cuda()
    net = DNL_3DCNN(4).cuda(2)
    #input = torch.randn(1, 1, 7, 320, 180).cuda()
    print(net)
    #summary(net,(1, 1, 7, 320, 180))


    # from thop import profile
    # input = torch.randn(1, 1, 7, 320, 180).cuda()
    # flops, params = profile(net, inputs=(input,))
    # total = sum([param.nelement() for param in net.parameters()])
    # print('   Number of params: %.2fM' % (total / 1e6))
    # print('   Number of FLOPs: %.2fGFLOPs' % (flops / 1e9))


