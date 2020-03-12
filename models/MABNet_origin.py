from __future__ import print_function
import torch
import torch.nn as nn
import torch.utils.data
from torch.autograd import Variable
import torch.nn.functional as F
import math
from .submodule import *

class MABmodule_3d(nn.Module):
    def __init__(self, inp, oup, stride):
        super(MABmodule_3d, self).__init__()

        self.stride = stride
        assert stride in [1, 2]

        inc = inp//2
        ouc = oup//2

        if self.stride == 2:
            self.conv0 = nn.Sequential(
                        nn.Conv3d(inp, inp, kernel_size=3, stride=2, padding=1, dilation=1, groups=inp, bias=False),
                        nn.BatchNorm3d(inp),
                        convbn_3d(inp, ouc, ksize=1, stride=1, pad=0, dila=1)
                        )

            self.conv1 = nn.Sequential(
                        convbn_3d(inp, ouc, ksize=(3,1,1), stride=(2,1,1), pad=(1,0,0), dila=1),
                        nn.ReLU(inplace=True),

                        nn.Conv3d(ouc, ouc, kernel_size=(1,3,3), stride=(1,2,2), padding=(0,1,1), dilation=1, groups=ouc, bias=False),
                        nn.BatchNorm3d(ouc),
                        
                        )

            self.conv2 = nn.Sequential(
                        convbn_3d(inp, ouc//2, ksize=(3,1,1), stride=(2,1,1), pad=(2,0,0), dila=(2,1,1)),
                        nn.ReLU(inplace=True),

                        nn.Conv3d(ouc//2, ouc//2, kernel_size=(1,3,3), stride=(1,2,2), padding=(0,2,2), dilation=(1,2,2), groups=ouc//2, bias=False),
                        nn.BatchNorm3d(ouc//2),
                        
                        )
            self.lastconv = convbn_3d(3*ouc//2, ouc, ksize=1, stride=1, pad=0, dila=1)
  
        else :
            self.conv1 = nn.Sequential(
                        convbn_3d(inc, ouc, ksize=(3,1,1), stride=1, pad=(1,0,0), dila=1),
                        nn.ReLU(inplace=True),

                        nn.Conv3d(ouc, ouc, kernel_size=(1,3,3), stride=1, padding=(0,1,1), dilation=1, groups=ouc, bias=False),
                        nn.BatchNorm3d(ouc),
                        
                        )

            self.conv2 = nn.Sequential(
                        convbn_3d(inc, ouc//2, ksize=(3,1,1), stride=1, pad=(2,0,0), dila=(2,1,1)),
                        nn.ReLU(inplace=True),

                        nn.Conv3d(ouc//2, ouc//2, kernel_size=(1,3,3), stride=1, padding=(0,2,2), dilation=(1,2,2), groups=ouc//2, bias=False),
                        nn.BatchNorm3d(ouc//2),
                        
                        )
            self.lastconv = convbn_3d(3*ouc//2, ouc, ksize=1, stride=1, pad=0, dila=1)

    def forward(self, x):
        x1 = x[:, :(x.shape[1]//2), :, :]
        x2 = x[:, (x.shape[1]//2):, :, :]

        if self.stride == 2 :
            out1 = torch.cat((self.conv1(x), self.conv2(x)),1)
            out = torch.cat((self.conv0(x), self.lastconv(out1)),1)
        else:
            out1 = torch.cat((self.conv1(x2), self.conv2(x2)),1)
            out = torch.cat((x1, self.lastconv(out1)),1)

        out = channel_shuffle_3d(out, 2)

        return out

class hourglass(nn.Module):
    def __init__(self, inp):
        super(hourglass, self).__init__()
        
        self.conv1_1 = convbn_3d(inp, inp, ksize=3, stride=2, pad=1, dila=1)
        self.conv1_2 = convbn_3d(inp, inp, ksize=3, stride=2, pad=2, dila=2)


        self.conv2_1 = MABmodule_3d(inp, inp, stride=1)
        self.conv2_2 = MABmodule_3d(inp, inp, stride=1)

        self.conv3_1 = convbn_3d(inp, inp//2, ksize=3, stride=2, pad=1, dila=1)
        self.conv3_2 = convbn_3d(inp, inp//2, ksize=3, stride=2, pad=2, dila=2)
        self.conv3_3 = convbn_3d(inp, inp//2, ksize=3, stride=2, pad=1, dila=1)
        self.conv3_4 = convbn_3d(inp, inp//2, ksize=3, stride=2, pad=2, dila=2)

        self.conv4 = MABmodule_3d(inp*2, inp*2, stride=1)

        self.conv5 = nn.Sequential(nn.ConvTranspose3d(inp*2, inp*2, kernel_size=3, padding=1, output_padding=1, stride=2, bias=False),
                                   nn.BatchNorm3d(inp*2))

        self.conv6 = nn.Sequential(nn.ConvTranspose3d(inp*2, inp, kernel_size=3, padding=1, output_padding=1, stride=2, bias=False),
                                   nn.BatchNorm3d(inp))

    def forward(self, x):
        
        out1 = self.conv1_1(x)
        out2 = self.conv1_2(x)
        out1 = F.relu(out1, inplace=True)
        out2 = F.relu(out2, inplace=True)

        pre1 = self.conv2_1(out1)
        pre2 = self.conv2_2(out2)
        
        out11 = self.conv3_1(pre1)
        out11 = F.relu(out11, inplace=True)
        out12 = self.conv3_2(pre1)
        out12 = F.relu(out12, inplace=True)
        out21 = self.conv3_3(pre2)
        out21 = F.relu(out21, inplace=True)
        out22 = self.conv3_4(pre2)
        out22 = F.relu(out22, inplace=True)

        out = torch.cat((out11,out12,out21,out22),1)
        out = self.conv4(out)

        out = self.conv5(out)
        pre = torch.cat((pre1,pre2),1)
        post = F.relu(out+pre, inplace=True)

        out  = self.conv6(post)  

        return out

class MABNet_origin(nn.Module):
    def __init__(self, maxdisp):
        super(MABNet_origin, self).__init__()
        self.maxdisp = maxdisp

        self.feature_extraction = feature_extraction_origin()

        self.dres0 = nn.Sequential(convbn_3d(64, 32, 3, 1, 1, 1),
                                   nn.ReLU(inplace=True),
                                   convbn_3d(32, 32, 3, 1, 1, 1),
                                   nn.ReLU(inplace=True))

        self.dres1 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1, 1),
                                   nn.ReLU(inplace=True),
                                   convbn_3d(32, 32, 3, 1, 1, 1)) 

        self.dres2 = hourglass(32)

        self.dres3 = hourglass(32)

        self.dres4 = hourglass(32)

        self.classif1 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1, 1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv3d(32, 1, kernel_size=3, padding=1, stride=1,bias=False))
                                      
        self.classif2 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1, 1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv3d(32, 1, kernel_size=3, padding=1, stride=1,bias=False))

        self.classif3 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1, 1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv3d(32, 1, kernel_size=3, padding=1, stride=1,bias=False))

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

        out2 = self.dres3(out1) 
        out2 = out2+cost0

        out3 = self.dres4(out2) 
        out3 = out3+cost0

        cost1 = self.classif1(out1)
        cost2 = self.classif2(out2) + cost1
        cost3 = self.classif3(out3) + cost2

        if self.training:
            cost1 = F.upsample(cost1, [self.maxdisp,left.size()[2],left.size()[3]], mode='trilinear')
            cost2 = F.upsample(cost2, [self.maxdisp,left.size()[2],left.size()[3]], mode='trilinear')
        
            cost1 = torch.squeeze(cost1,1)
            pred1 = F.softmax(cost1,dim=1)
            pred1 = disparityregression(self.maxdisp)(pred1)
        
            cost2 = torch.squeeze(cost2,1)
            pred2 = F.softmax(cost2,dim=1)
            pred2 = disparityregression(self.maxdisp)(pred2)
        
        cost3 = F.upsample(cost3, [self.maxdisp,left.size()[2],left.size()[3]], mode='trilinear')
        cost3 = torch.squeeze(cost3,1)
        pred3 = F.softmax(cost3,dim=1)
        pred3 = disparityregression(self.maxdisp)(pred3)

        if self.training:
            return pred1, pred2, pred3
        else:
            return pred3
