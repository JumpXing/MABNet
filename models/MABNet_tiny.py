from __future__ import print_function
import torch
import torch.nn as nn
import torch.utils.data
from torch.autograd import Variable
import torch.nn.functional as F
import math
from .submodule import *
from .MABNet_origin import hourglass

class MABNet_tiny(nn.Module):
    def __init__(self, maxdisp):
        super(MABNet_tiny, self).__init__()
        self.maxdisp = maxdisp

        self.feature_extraction = feature_extraction_tiny()

        self.dres0 = nn.Sequential(convbn_3d(16, 8, 3, 1, 1, 1),
                                   nn.ReLU(inplace=True),
                                   convbn_3d(8, 8, 3, 1, 1, 1),
                                   nn.ReLU(inplace=True))

        self.dres1 = nn.Sequential(convbn_3d(8, 8, 3, 1, 1, 1),
                                   nn.ReLU(inplace=True),
                                   convbn_3d(8, 8, 3, 1, 1, 1)) 

        self.dres2 = hourglass(8)

        self.classif1 = nn.Sequential(convbn_3d(8, 8, 3, 1, 1, 1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv3d(8, 1, kernel_size=3, padding=1, stride=1,bias=False))
                                      

        #权值参数初始化
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1]*m.kernel_size[2] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()


    def forward(self, left, right):

        #feature extraction
        refimg_fea    = self.feature_extraction(left)
        targetimg_fea = self.feature_extraction(right)

        #building cost
        cost = Variable(torch.FloatTensor(refimg_fea.size()[0], refimg_fea.size()[1]*2, int(self.maxdisp/4),  refimg_fea.size()[2],  refimg_fea.size()[3]).zero_()).cuda()

        for i in range(int(self.maxdisp/4)):
            if i > 0 :
                cost[:, :refimg_fea.size()[1], i, :,i:]  = refimg_fea[:,:,:,i:]
                cost[:, refimg_fea.size()[1]:, i, :,i:]  = targetimg_fea[:,:,:,:-i] 
            else:
                cost[:, :refimg_fea.size()[1], i, :,:]   = refimg_fea
                cost[:, refimg_fea.size()[1]:, i, :,:]   = targetimg_fea

        cost = cost.contiguous()

        #disparity regression
        cost0 = self.dres0(cost)
        cost0 = self.dres1(cost0) + cost0

        out1 = self.dres2(cost0) 
        out1 = out1+cost0

        cost1 = self.classif1(out1)

        cost1 = F.upsample(cost1, [self.maxdisp,left.size()[2],left.size()[3]], mode='trilinear')
        
        cost1 = torch.squeeze(cost1,1)
        pred1 = F.softmax(cost1,dim=1)
        pred1 = disparityregression(self.maxdisp)(pred1)
        
        return pred1
