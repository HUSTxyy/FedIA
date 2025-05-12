
import logging
import numpy as np

import torch
import torch.optim
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from dataset.dataset import my_collate

from utils.losses import dice_score
from utils.labelCorrect import CorrectLabel_threshold
from scipy.ndimage import label

def localtest(net, test_dataset, args):
    net.eval()
    test_loader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    
    val_Dice = 0
    count = 0  # np.zeros(shape=(7, 2))
    with torch.no_grad():
        for iter,batch in enumerate(test_loader):
            images,labels,name,index=batch
            images = images.to(dtype=torch.float32).cuda()
            # print("img",images.shape)
            labels = labels.to(dtype=torch.long).cuda()
        
            outputs = net(images)
            pred = outputs.argmax(dim=1)
            dice = dice_score(labels, pred)
            val_Dice+= dice
            # print(dice)
        
            count+= 1
           
       
        val_Dice = val_Dice/ count

    return val_Dice


class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return self.idxs[item], image, label



class LocalUpdate(object):
    def __init__(self, args, id, dataset):
        self.args = args
        self.id = id
        self.local_dataset = dataset
        # self.local_datasetTest = datasetTest

        logging.info(
            f'client{id} total: {len(self.local_dataset)}')
        self.ldr_train = DataLoader(
            self.local_dataset, batch_size=self.args.batch_size, shuffle=True, num_workers=0, collate_fn=my_collate)
        self.epoch = 0
        self.iter_num = 0
        self.lr = self.args.base_lr


    def train_local(self, net, writer,round,correct,thresh):
        net.train()
        
        # set the optimizer
        self.optimizer = torch.optim.Adam(
            net.parameters(), lr=self.lr, betas=(0.9, 0.999), weight_decay=5e-4)

        # train and update
        epoch_loss = []
        for epoch in range(self.args.local_ep):
            batch_loss = []
            batch_anR=[]
            for iter,batch in enumerate(self.ldr_train):#这个里面的数据到底改没改，数据加载是怎么说
                    
                images = torch.from_numpy(batch['image']).to(device=self.args.device, dtype=torch.float32)
                # print("img",images.shape)
                labels = torch.from_numpy(batch['label']).to(device=self.args.device, dtype=torch.long)
            # for (images, labels, _) in self.ldr_train:########要改
            #     images, labels = images.to(self.args.device), labels.cuda().to(self.args.device)
                index=batch['index']
                logits = net(images)

                anRate=1.0
                if round==30:
                    compo=logits.argmax(dim=1)
                    preL,preN=label(compo.cpu().numpy())
                    suL,suN=label(labels.cpu().numpy())
                    anRate=suN*1.0/preN
                    batch_anR.append(anRate)

                if round>=30 and correct!=0 and round==correct:
                # if round==25 and self.id!=3:#15best,应该是上一轮的校正下一轮的？
                    new_labels=CorrectLabel_threshold(batch['label'], logits.cpu().detach().numpy(),conf_threshold=thresh)
                    self.ldr_train.dataset.reset_labels(new_labels,index)
                # if round==27 and self.id!=3:#15best,应该是上一轮的校正下一轮的？
                #     new_labels=CorrectLabel_threshold(batch['label'], logits.cpu().detach().numpy(),conf_threshold=0.85)
                #     self.ldr_train.dataset.reset_labels(new_labels,index)
              
                loss = 1-dice_score(logits[:,1,:,:], labels)#####可能出问题

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                batch_loss.append(loss.item())
                writer.add_scalar(
                    f'client{self.id}/loss_train', loss.item(), self.iter_num)
                self.iter_num += 1
            self.epoch = self.epoch + 1
            epoch_loss.append(np.array(batch_loss).mean())

            anR10=np.array(batch_anR).mean()
        # dice = localtest(net, self.local_datasetTest, self.args)
        return net.state_dict(), np.array(epoch_loss).mean(), anR10
