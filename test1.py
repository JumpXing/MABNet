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
import skimage
import skimage.io
import skimage.transform
import numpy as np
import time
import math
from dataloader import KITTIloader2015 as ls
from dataloader import KITTILoader as DA

from models import *

parser = argparse.ArgumentParser(description='ShuffleStereo')
parser.add_argument('--maxdisp', type=int ,default=192,
                    help='maxium disparity')
parser.add_argument('--model', default='ShuffleStereo8',
                    help='select model')
parser.add_argument('--datapath', default='/home/jump/dataset/kitti2015/training/',
                    help='datapath')
parser.add_argument('--loadmodel', default='./trained/checkpoint_12.tar',
                    help='load model')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed_all(args.seed)

test_left_img1, test_right_img1, test_left_disp1, test_left_img, test_right_img, test_left_disp = ls.dataloader(args.datapath)

TestImgLoader = torch.utils.data.DataLoader(
         DA.myImageFloder(test_left_img,test_right_img,test_left_disp, False), 
         batch_size= 1, shuffle= False, num_workers= 4, drop_last=False)

if args.model == 'ShuffleStereo8':
    model = MABNet_origin(args.maxdisp)
elif args.model == 'ShuffleStereo16':
    model = ShuffleStereo16(args.maxdisp)
else:
    print('no model')

if args.cuda:
    model = nn.DataParallel(model)
    model.cuda()

if args.loadmodel is not None:
    state_dict = torch.load(args.loadmodel)
    model.load_state_dict(state_dict['state_dict'])
    print(state_dict['epoch'])
    print(state_dict['train_loss'])
    #print(state_dict['test_loss'])

#if isinstance(model,torch.nn.DataParallel):
#    model = model.module

print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))
#print(model) #model是直接打印模型。 #model.modules()是遍历所有层，model.children()仅是遍历当前层
'''for k, m in enumerate(model.modules()):
    if k == 213:
        print(k,m)'''
#print(model.parameters().feature_extraction.firstconv)
#print(model.state_dict().keys())



# x=0
# for n,p in model.state_dict().items():
#     x+=1
#     #if x<6:
#     print(n,p)

def test(imgL,imgR,disp_true):
        model.eval()
       
        imgL   = Variable(torch.FloatTensor(imgL))
        imgR   = Variable(torch.FloatTensor(imgR)) 
        if args.cuda:
            imgL, imgR = imgL.cuda(), imgR.cuda()

        with torch.no_grad():
            output = model(imgL,imgR)

        pred_disp = output.data.cpu()
        
        #print(str(n)+'pred_disp.shape:',pred_disp.shape)
        #temp = output.cpu()
        #temp = temp.detach().numpy()
        #print(temp.shape)
        #skimage.io.imsave('./png/'+str(n)+test_left_img[n].split('/')[-1], (temp[0, :, :]*256).astype('uint16'))

        #computing 3-px error#
        true_disp = disp_true
        index = np.argwhere(true_disp>0)
        disp_true[index[0][:], index[1][:], index[2][:]] = np.abs(true_disp[index[0][:], index[1][:], index[2][:]]-pred_disp[index[0][:], index[1][:], index[2][:]])
        correct = (disp_true[index[0][:], index[1][:], index[2][:]] < 3)|(disp_true[index[0][:], index[1][:], index[2][:]] < true_disp[index[0][:], index[1][:], index[2][:]]*0.05)      
        torch.cuda.empty_cache()

        return 1-(float(torch.sum(correct))/float(len(index[0])))

def main():
    max_acc=0
    start_full_time = time.time()
    
    total_test_loss = 0

    ## Test ##
    for batch_idx, (imgL, imgR, disp_L) in enumerate(TestImgLoader):
        
        test_loss = test(imgL,imgR, disp_L)
        print('Iter %d 3-px error in val = %.3f' %(batch_idx, test_loss*100))
        total_test_loss += test_loss

    max_acc = total_test_loss/len(TestImgLoader)*100
    print('total test error = %.3f' %(max_acc))	
    print('full test time = %.2f s' %(time.time() - start_full_time))

if __name__ == '__main__':
   main()
