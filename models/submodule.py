from __future__ import print_function
import torch
import torch.nn as nn
import torch.utils.data
from torch.autograd import Variable
import torch.nn.functional as F
import math
import numpy as np
#import skimage
#import skimage.io
#import skimage.transform

def convbn(inp, oup, ksize, stride, pad, dila, groups):

    return nn.Sequential(nn.Conv2d(inp, oup, kernel_size=ksize, stride=stride, padding=dila if dila > 1 else pad, dilation=dila, groups=groups, bias=False),
                         nn.BatchNorm2d(oup))

def convbn_3d(inp, oup, ksize, stride, pad, dila):

    return nn.Sequential(nn.Conv3d(inp, oup, kernel_size=ksize, stride=stride, padding=pad, dilation=dila, bias=False),
                         nn.BatchNorm3d(oup))

def channel_shuffle(x, groups):
    batchsize, num_channels, height, width = x.data.size()

    channels_per_group = num_channels // groups
    
    # reshape
    x = x.view(batchsize, groups, channels_per_group, height, width)

    x = torch.transpose(x, 1, 2).contiguous()

    # flatten
    x = x.view(batchsize, -1, height, width)

    return x

def channel_shuffle_3d(x, groups):
    batchsize, num_channels, disparity, height, width = x.data.size()

    channels_per_group = num_channels // groups
    
    # reshape
    x = x.view(batchsize, groups, channels_per_group, disparity, height, width)

    x = torch.transpose(x, 1, 2).contiguous()

    # flatten
    x = x.view(batchsize, -1, disparity, height, width)

    return x

class MABmodule(nn.Module):
    def __init__(self, inp, oup, stride, factor, benchmodel):
        super(MABmodule, self).__init__()
        self.benchmodel = benchmodel
        
        inc = inp//2
        ouc = oup//2
        
        if self.benchmodel == 1:
        	self.banch2 = nn.Sequential(
                # pw
                convbn(inc, ouc*factor, 1, 1, 0, 1, 1),
                nn.ReLU(inplace=True),

                # dw
                convbn(ouc*factor, ouc*factor, 3, stride, 1, 1, ouc*factor),
            )
        	self.banch3 = nn.Sequential(
                # pw
                convbn(inc, ouc*factor//2, 1, 1, 0, 1, 1),
                nn.ReLU(inplace=True),

                # dw
                convbn(ouc*factor//2, ouc*factor//2, 3, stride, 2, 2, ouc*factor//2),
            )
        	self.banch4 = nn.Sequential(
                # pw
                convbn(inc, ouc*factor//4, 1, 1, 0, 1, 1),
                nn.ReLU(inplace=True),

                # dw
                convbn(ouc*factor//4, ouc*factor//4, 3, stride, 4, 4, ouc*factor//4),
            )
        	self.lastconv = convbn(7*ouc*factor//4, ouc, 1, 1, 0, 1, 1)                  
        else:                  
            self.banch1 = nn.Sequential(
                # dw
                convbn(inp, inp, 3, stride, 1, 1, inp),
                # pw-linear
                convbn(inp, ouc, 1, 1, 0, 1, 1)
            )        
    
            self.banch2 = nn.Sequential(
                # pw
                convbn(inp, ouc*factor, 1, 1, 0, 1, 1),
                nn.ReLU(inplace=True),
                
                # dw
                convbn(ouc*factor, ouc*factor, 3, stride, 1, 1, ouc*factor),
            )

            self.banch3 = nn.Sequential(
                # pw
                convbn(inp, ouc*factor//2, 1, 1, 0, 1, 1),
                nn.ReLU(inplace=True),
                
                # dw
                convbn(ouc*factor//2, ouc*factor//2, 3, stride, 2, 2, ouc*factor//2),
            )
            self.banch4 = nn.Sequential(
                # pw
                convbn(inp, ouc*factor//4, 1, 1, 0, 1, 1),
                nn.ReLU(inplace=True),
                
                # dw
                convbn(ouc*factor//4, ouc*factor//4, 3, stride, 4, 4, ouc*factor//4),
            )
            self.lastconv = convbn(7*ouc*factor//4, ouc, 1, 1, 0, 1, 1)
            
    def forward(self, x):
        if 1==self.benchmodel:
            x1 = x[:, :(x.shape[1]//2), :, :]
            x2 = x[:, (x.shape[1]//2):, :, :]
            out1 = torch.cat((self.banch2(x2),self.banch3(x2),self.banch4(x2)),1)
            out = torch.cat((x1,self.lastconv(out1)),1)
        elif 2==self.benchmodel:
            out1 = torch.cat((self.banch2(x),self.banch3(x),self.banch4(x)),1)
            out = torch.cat((self.banch1(x), self.lastconv(out1)),1)

        return channel_shuffle(out, 2)

class disparityregression(nn.Module):
    def __init__(self, maxdisp):
        super(disparityregression, self).__init__()
        self.disp = Variable(torch.Tensor(np.reshape(np.array(range(maxdisp)),[1,maxdisp,1,1])).cuda(), requires_grad=False)

    def forward(self, x):
        disp = self.disp.repeat(x.size()[0],1,x.size()[2],x.size()[3])
        out = torch.sum(x*disp,1)
        return out

class feature_extraction_origin(nn.Module):
    def __init__(self):
        super(feature_extraction_origin, self).__init__()

        self.firstconv = nn.Sequential(convbn(3, 32, 3, 2, 1, 1, 1),
                                       nn.ReLU(inplace=True),
                                       convbn(32, 32, 3, 1, 1, 1, 1),
                                       nn.ReLU(inplace=True),
                                       convbn(32, 32, 3, 1, 1, 1, 1),
                                       nn.ReLU(inplace=True))
        
        self.inplanes = 32

        self.layer1 = self._make_group(MABmodule, 32, 3, 1, 1)
        self.layer2 = self._make_group(MABmodule, 64, 16, 2, 1)
        self.layer3 = self._make_group(MABmodule, 128, 3, 1, 1)
        self.layer4 = self._make_group(MABmodule, 128, 3, 1, 1)

        self.fusionconv = nn.Sequential(convbn(320, 128, 3, 1, 1, 1, 1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv2d(128, 32, kernel_size=1, stride = 1, padding=0, bias=False))

    def _make_group(self, block, planes, blocks, stride, factor):
        layers = []
        
        if stride != 1 or self.inplanes != planes:
            layers.append(block(self.inplanes, planes, stride, factor, 2))
        else:
            layers.append(block(self.inplanes, planes, stride, factor, 1))
            
        self.inplanes = planes

        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, 1, factor, 1))

        return nn.Sequential(*layers)

    def forward(self, x):
        output  = self.firstconv(x)    
        output  = self.layer1(output)
        output2 = self.layer2(output)
        output3 = self.layer3(output2)
        output4 = self.layer4(output3)

        output_feature = torch.cat((output2, output3, output4), 1)
        output_feature = self.fusionconv(output_feature)

        return output_feature

class feature_extraction_tiny(nn.Module):
    def __init__(self):
        super(feature_extraction_tiny, self).__init__()

        self.firstconv = nn.Sequential(convbn(3, 8, 3, 2, 1, 1, 1),
                                       nn.ReLU(inplace=True),
                                       convbn(8, 8, 3, 1, 1, 1, 1),
                                       nn.ReLU(inplace=True),
                                       convbn(8, 8, 3, 1, 1, 1, 1),
                                       nn.ReLU(inplace=True))
        
        self.inplanes = 8

        self.layer1 = self._make_group(MABmodule, 8,4, 1, 1)
        self.layer2 = self._make_group(MABmodule, 16, 8, 2, 1)
        self.layer3 = self._make_group(MABmodule, 32, 4, 1, 1)

        self.fusionconv = nn.Sequential(convbn(48, 16, 3, 1, 1, 1, 1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv2d(16, 8, kernel_size=1, stride = 1, padding=0, bias=False))

    def _make_group(self, block, planes, blocks, stride, factor):
        layers = []
        
        if stride != 1 or self.inplanes != planes:
            layers.append(block(self.inplanes, planes, stride, factor, 2))
        else:
            layers.append(block(self.inplanes, planes, stride, factor, 1))
            
        self.inplanes = planes

        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, 1, factor, 1))

        return nn.Sequential(*layers)

    def forward(self, x):
        output  = self.firstconv(x)    
        output  = self.layer1(output)
        output2 = self.layer2(output)
        output3 = self.layer3(output2)

        output_feature = torch.cat((output2, output3), 1)
        output_feature = self.fusionconv(output_feature)

        return output_feature


