import os
import copy
import logging
import numpy as np

from collections import Counter

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from scipy.optimize import curve_fit
import math
from utils.options import args_parser
from utils.local_training import LocalUpdate, localtest
from utils.FedAvg import FedAvg
from utils.utils import set_seed, set_output_files
from utils.losses import dice_score
from dataset.all_datasets import MS
from model.model_UNet import UNet
np.set_printoptions(threshold=np.inf)

"""
Major framework of noise FL
"""
def curve_func(x, a, b):
    return a *x+b


def fit(func, x, y):
    popt, pcov = curve_fit(func, x, y)
    return tuple(popt)



def label_update_epoch(ydata_fit):
    xdata_fit = np.linspace(0, len(ydata_fit)-1, len(ydata_fit))
    # print(xdata_fit)
    # print(ydata_fit)
    a, b = fit(curve_func, xdata_fit, ydata_fit)

    return a,b  # , a, b, c

if __name__ == '__main__':
    args = args_parser()
    args.num_users = args.n_clients
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    args.device = "cuda" if torch.cuda.is_available() else "cpu"


    # ------------------------------ deterministic or not ------------------------------
    if args.deterministic:
        cudnn.benchmark = False
        cudnn.deterministic = True
        set_seed(args.seed)

    # ------------------------------ output files ------------------------------
    writer, models_dir = set_output_files(args)

    # ------------------------------ dataset ------------------------------
    data={}
    val_data={}
    user_id = list(range(args.n_clients))
    for id in user_id:
        data[id] = MS(args.data_dir, 'client'+str(id)+'.txt', id)
    val_data = MS(args.data_dir, 'test.txt', None)
        # test_data = MS(args.data_dir, 'central200test.txt', None)

    # --------------------- Build Models ---------------------------
    netglob = UNet(args)
    netglob = netglob.to(args.device)
   
    #netglob.load_state_dict(torch.load("/home/xyy/data2D/fromStart/outputs_MS/Fed_1/modelsold/model_30.pth", map_location=torch.device('cpu')))
    trainer_locals = []
    for id in user_id:
        trainer_locals.append(LocalUpdate(
            args, id, copy.deepcopy(data[id])))

    # ------------------------------ begin training ------------------------------
    set_seed(args.seed)
    logging.info("\n ---------------------begin training---------------------")
    best_performance = 0.

    # ------------------------ Stage 1: ------------------------ 
    DICE = []
    flag=[]
    flag.append(0)
    flag.append(0)
    flag.append(0)
    flag.append(0)
    a=[]
    a.append(1)
    a.append(1)
    a.append(1)
    a.append(1)
    b=[]
    b.append(1)
    b.append(1)
    b.append(1)
    b.append(1)
    correct=[]
    correct.append(0)
    correct.append(0)
    correct.append(0)
    correct.append(0)
    thre=[]
    thre.append(1)
    thre.append(1)
    thre.append(1)
    thre.append(1)
    IoU_npl_dict = {}
    for i in range(4):
        IoU_npl_dict[i] = []
    an_R=[]
    for rnd in range(args.rounds):
        w_locals, loss_locals, dice_locals = [], [], []

        for idx in user_id:  # training over the subset
            

            local = trainer_locals[idx]  
            w_local, loss_local,anRate = local.train_local(
            net=copy.deepcopy(netglob).to(args.device), writer=writer,round=rnd,correct=correct[idx],thresh=thre[idx])
            
            if rnd==30:
                an_R.append(anRate)

            # store every updated model
            w_locals.append(copy.deepcopy(w_local))
            loss_locals.append(copy.deepcopy(loss_local))
        
            assert len(w_locals) == len(loss_locals) == idx+1
            writer.add_scalar(
                    f'client{idx}/epoch_loss_train', loss_local, rnd)
           
            
            IoU_npl_dict[idx].append((1.0-loss_locals[idx])/(1.0+loss_locals[idx]))
        
            if rnd==30:
                # if an_R[idx]>0.5:
                #     correct[idx]=300
                #     thre[idx]=1
                #     print(idx)
                #     print("no correct")
                #     flag[idx]=1
                # else:
                tmpa,tmpb=label_update_epoch(IoU_npl_dict[idx])
                a[idx]=tmpa
                b[idx]=tmpb
                thre[idx]=0.8#min((an_R[idx]+0.4),0.8)
                print(thre[idx])
            if rnd>30 and correct[idx]!=300 and flag[idx]!=1:
                line=a[idx]*rnd+b[idx]
                if line-IoU_npl_dict[idx][rnd]>0.03:
                    correct[idx]=rnd+1##后面的数据也要加上？
                    print(correct[idx])
                    flag[idx]=1
                    



        

        if rnd<=30  or (flag[0]==1 and flag[1]==1 and flag[2]==1 and flag[3]==1):
            dict_len = [626,591,570,646]

        
        
        
        else:
            print(an_R)
            dict_len=[math.exp(1.0/loss_locals[0]*an_R[0]),math.exp(1.0/loss_locals[1]*an_R[1]),math.exp(1.0/loss_locals[2]*an_R[2]),math.exp(1.0/loss_locals[3]*an_R[3])]
            # dict_len = [math.exp(1.0/loss_locals[0]),math.exp(1.0/loss_locals[1]),math.exp(1.0/loss_locals[2]),math.exp(1.0/loss_locals[3])]

        # else:
        #     dict_len = [463,370,308,389]
        
        w_glob_fl = FedAvg(w_locals, dict_len)
        netglob.load_state_dict(copy.deepcopy(w_glob_fl))

        diceGlobal = localtest(netglob, val_data, args)
       
        writer.add_scalar(f'testglob/dice', diceGlobal, rnd)
       

      


        
        # print(dice)
        
        logging.info(
            "******** round: %d, dice: %.4f ********" % (rnd, diceGlobal))
       
        # save model
        if diceGlobal > best_performance:
            best_performance = diceGlobal
        torch.save(netglob.state_dict(),  models_dir+f'/model_{rnd}.pth')
        logging.info(f'best dice: {best_performance}, now dice: {diceGlobal}')
        logging.info('\n')
    # torch.save(netglob.state_dict(),  models_dir+f'/model_{rnd}.pth')

    # DICE = np.array(DICE)
    # logging.info("last:")
    # logging.info(DICE[-10:].mean())
    # logging.info("best:")
    # logging.info(DICE.max())

    torch.cuda.empty_cache()

