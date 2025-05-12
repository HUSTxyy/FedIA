import argparse
import os, sys

sys.path.append("..")

import torch
import torch.nn as nn
from torch.utils import data
import numpy as np
import pickle
import torch.optim as optim
import scipy.misc
import torch.backends.cudnn as cudnn
import torch.nn.functional as F

from scipy.ndimage.filters import gaussian_filter
from skimage import io

# from tqdm import tqdm
import os.path as osp


from sklearn import metrics
import nibabel as nib
from utils.local_training import localtest
from dataset.all_datasets import MS
from utils.FedAvg import FedAvg



from torch.utils.data import DataLoader
from model.model_UNet import UNet

def get_arguments():

    parser = argparse.ArgumentParser()

    # system setting
    parser.add_argument('--deterministic', type=int,  default=1,
                        help='whether use deterministic training')
    parser.add_argument('--seed', type=int,  default=0, help='random seed')
    parser.add_argument('--gpu', type=str,  default='3', help='GPU to use')

    # basic setting
    parser.add_argument('--exp', type=str,
                        default='Fed', help='experiment name')
    parser.add_argument('--dataset', type=str,
                        default='MS', help='dataset name')#########
    parser.add_argument('--model', type=str,
                        default='UNet2D', help='model name')############
    parser.add_argument('--batch_size', type=int,
                        default=1, help='batch_size per gpu')
    parser.add_argument('--base_lr', type=float,  default=1e-4,
                        help='base learning rate')
    parser.add_argument('--bilinear', default=True, help='Use bilinear upsampling')
    parser.add_argument('--classes', '-c', type=int, default=2, help='Number of classes')

    parser.add_argument("--data_dir", type=str, default='/home/xyy/data2D/')
    return parser
def main():
    parser = get_arguments()
    args = parser.parse_args()
    model=UNet(args)
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    # model.load_state_dict(torch.load('../outputs_MS/Fed_1/models/model_134.pth', map_location=torch.device('cpu')))
#    dict_len = [1,2,3,4000]
#    model_stage1=[]
#    model_stage1.append(torch.load('/home/xyy/Brain2D/data2D/client0In4/outputs_MS/Fed_1/models/model_18.pth', map_location=torch.device('cpu')))
#    model_stage1.append(torch.load('/home/xyy/Brain2D/data2D/client1In6/outputs_MS/Fed_1/models/model_12.pth', map_location=torch.device('cpu')))
#    model_stage1.append(torch.load('/home/xyy/Brain2D/data2D/client2In8/outputs_MS/Fed_1/models/model_53.pth', map_location=torch.device('cpu')))
#    model_stage1.append(torch.load('/home/xyy/Brain2D/data2D/client3_200/outputs_MS/Fed_1/models/model_132.pth', map_location=torch.device('cpu')))
#    model_fl = FedAvg(model_stage1, dict_len)
    model.load_state_dict(torch.load('../outputs_MS/Fed_1/models/model_33.pth', map_location=torch.device('cpu')))
    test_set = MS(args.data_dir, 'central200test.txt', None)
    dice = localtest(model, test_set, args)
    print(dice)
   
      
if __name__ == '__main__':
    main()





