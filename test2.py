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
import numpy as np
import time
import math
from dataloader import listflowfile as lt
from dataloader import SecenFlowLoader as DA
from models import *

parser = argparse.ArgumentParser(description='MABNet')
parser.add_argument('--maxdisp', type=int ,default=192,
                    help='maxium disparity')
parser.add_argument('--model', default='MABNet_origin',
                    help='select model')
parser.add_argument('--datapath', default='/home/jump/dataset/sceneflow',
                    help='datapath')
parser.add_argument('--loadmodel', default= './trained_origin/checkpoint_10.tar',
                    help='load model')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

# set gpu id used
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed_all(args.seed)

all_left_img, all_right_img, all_left_disp, test_left_img, test_right_img, test_left_disp = lt.dataloader(args.datapath)

TestImgLoader = torch.utils.data.DataLoader(
         DA.myImageFloder(test_left_img,test_right_img,test_left_disp, False), 
         batch_size= 8, shuffle= False, num_workers= 4, drop_last=False)


if args.model == 'MABNet_origin':
    model = MABNet_origin(args.maxdisp)
elif args.model == 'MABNet_tiny':
    model = MABNet_tiny(args.maxdisp)
else:
    print('no model')

if args.cuda:
    model = nn.DataParallel(model)
    model.cuda()

if args.loadmodel is not None:
    state_dict = torch.load(args.loadmodel)
    model.load_state_dict(state_dict['state_dict'])

print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))

def test(imgL,imgR,disp_true):
        model.eval()
        imgL   = Variable(torch.FloatTensor(imgL))
        imgR   = Variable(torch.FloatTensor(imgR))   
        if args.cuda:
            imgL, imgR = imgL.cuda(), imgR.cuda()

        #---------
        mask = disp_true < 192
        #----

        with torch.no_grad():
            output3 = model(imgL,imgR)

        output = torch.squeeze(output3.data.cpu(),1)[:,4:,:]

        if len(disp_true[mask])==0:
           loss = 0
        else:
           loss = torch.mean(torch.abs(output[mask]-disp_true[mask]))  # end-point-error

        return loss

def main():

	start_full_time = time.time()
	#------------- TEST ------------------------------------------------------------
	total_test_loss = 0
	for batch_idx, (imgL, imgR, disp_L) in enumerate(TestImgLoader):
	       test_loss = test(imgL,imgR, disp_L)
	       print('Iter %d test loss = %.3f' %(batch_idx, test_loss))
	       total_test_loss += test_loss

	print('total test loss = %.3f' %(total_test_loss/len(TestImgLoader)))
	print('full test time = %.2f s' %(time.time() - start_full_time))

if __name__ == '__main__':
   main()
    
