from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
import torch.nn.functional as F
from PIL import Image, ImageOps
import skimage
import skimage.io
import skimage.transform
import numpy as np
import time
import math
from utils import preprocess 
from models import *

# 2012 data /home/jump/dataset/kitti2012/testing/

parser = argparse.ArgumentParser(description='MABNet: Multi-branch Adjustable Bottleneck Network')
parser.add_argument('--KITTI', default='2012',
                    help='KITTI version')
parser.add_argument('--datapath', default='/home/jump/dataset/kitti2012/testing/',
                    help='select model')
parser.add_argument('--loadmodel', default='./trained_tiny_2012/finetune_586.tar',
                    help='loading model')
parser.add_argument('--model', default='MABNet_tiny',
                    help='select model')
parser.add_argument('--maxdisp', type=int, default=192,
                    help='maxium disparity')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
args = parser.parse_args()

args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

if args.KITTI == '2015':
   from dataloader import KITTI_submission_loader as DA
else:
   from dataloader import KITTI_submission_loader2012 as DA  


test_left_img, test_right_img = DA.dataloader(args.datapath)

if args.model == 'MABNet_origin':
    model = MABNet_origin(args.maxdisp)
elif args.model == 'MABNet_tiny':
    model = MABNet_tiny(args.maxdisp)
else:
    print('no model')

model = nn.DataParallel(model, device_ids=[0])
model.cuda()

if args.loadmodel is not None:
    state_dict = torch.load(args.loadmodel)
    model.load_state_dict(state_dict['state_dict'])

print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))

def test(imgL,imgR):
        model.eval()

        if args.cuda:
           imgL = torch.FloatTensor(imgL).cuda()
           imgR = torch.FloatTensor(imgR).cuda()     

        imgL, imgR= Variable(imgL), Variable(imgR)

        with torch.no_grad():
            output = model(imgL,imgR)
        output = torch.squeeze(output)
        pred_disp = output.data.cpu().numpy()

        return pred_disp


def main():
   processed = preprocess.get_transform(augment=False)

   for inx in range(len(test_left_img)):

       imgL_o = Image.open(test_left_img[inx]).convert('RGB')
       imgR_o = Image.open(test_right_img[inx]).convert('RGB')
       imgL_o = processed(imgL_o).numpy()
       imgR_o = processed(imgR_o).numpy()
       imgL = np.reshape(imgL_o,[1,3,imgL_o.shape[1],imgL_o.shape[2]])
       imgR = np.reshape(imgR_o,[1,3,imgR_o.shape[1],imgR_o.shape[2]])

       # pad to (384, 1248)
       top_pad = 384-imgL.shape[2]
       left_pad = 1248-imgL.shape[3]
       imgL = np.lib.pad(imgL,((0,0),(0,0),(top_pad,0),(0,left_pad)),mode='constant',constant_values=0)
       imgR = np.lib.pad(imgR,((0,0),(0,0),(top_pad,0),(0,left_pad)),mode='constant',constant_values=0)

       start_time = time.time()
       pred_disp = test(imgL,imgR)
       print('time = %.2f' %(time.time() - start_time))

       img = pred_disp[top_pad:,:-left_pad]
       skimage.io.imsave('./disp_0/'+test_left_img[inx].split('/')[-1],(img*256).astype('uint16'))

if __name__ == '__main__':
   main()






