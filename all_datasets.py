
import os
import numpy as np
from PIL import Image
import pandas as pd
import os.path as osp
import torch
from torch.utils.data import Dataset


# class MS(Dataset):
#     def __init__(self, root, list_path, mode, transform=None):
#         self.root = root
#         self.mode = mode
#         assert self.mode in ["train", "test"]
#         self.transform = transform

#         self.list_path = list_path
#         self.img_ids = [i_id.strip().split() for i_id in open(self.root + self.list_path)]



#         csv_file = os.path.join(self.root, self.mode+".csv")
#         self.file = pd.read_csv(csv_file)
#         self.images = self.file["image"].values
#         self.labels = self.file.iloc[:, 1:].values.astype("int")
#         self.targets = np.argmax(self.labels, axis=1)
#         self.n_classes = len(np.unique(self.targets))
#         assert self.n_classes == 2

#     def __len__(self):
#         return len(self.images)

#     def __getitem__(self, index):
#         image_path = os.path.join(
#             self.root, "ISIC_2019_Training_Input", self.images[index]+".jpg")
#         img = Image.open(image_path).convert("RGB")
#         img = self.transform(img)
#         label = self.targets[index]
#         return img, label

    

class MS(Dataset):
    def __init__(self, root, list_path, flag):
        self.root = root
        self.cache_label = []
        self.list_path = list_path
        self.img_ids = [i_id.strip().split() for i_id in open(self.root + self.list_path)]
        self.files = []
        for item in self.img_ids:
            # 切成2D的话，这块可以重新规划一下
            name=str(item)[2:-2]
            image_dwi = osp.join(self.root,'image/',name[:-4]+'_dwi.png')#######这个要改
            image_adc = osp.join(self.root,'image/',name[:-4]+'_adc.png')#######这个要改
            if flag==None:
                label_file = osp.join(self.root,'label/',name)###这个要改
               
            elif flag==0:
                label_file = osp.join(self.root,'2DlabelSingle/',name)###这个要改
              
            elif flag==1:
                label_file = osp.join(self.root,'2DlabelSingle/',name)###这个要改
               
            elif flag==2:
                label_file = osp.join(self.root,'2DlabelSingle/',name)###这个要改
             
            elif flag==3:
                label_file = osp.join(self.root,'2DlabelSingle/',name)###这个要改
              
            self.files.append({
                    "imagedwi": image_dwi,
                    "imageadc": image_adc,
                    "label": label_file,
                    "name": name,
                })
            self.cache_label.append(np.array(Image.open(label_file)))
#需不需要一个gt

    def __len__(self):
        return len(self.files)
    
    def reset_labels(self, new_labels,index):
        # print(self.cache_label.shape)   
        for pid in range(index.shape[0]):
            self.cache_label[index[pid]] = new_labels[pid]

    def __getitem__(self, index):
        datafiles = self.files[index]
        dwi = Image.open(datafiles["imagedwi"])
        adc = Image.open(datafiles["imageadc"])
       
        imagedwi = np.array(dwi)
        imageadc = np.array(adc)

        image=np.concatenate((imageadc[np.newaxis, :],imagedwi[np.newaxis, :]))

        
        name = datafiles["name"]
        # image = image[np.newaxis, :]
        

#不能把原始数据变了哇
        if self.cache_label is None:
            label = Image.open(datafiles["label"])
            label = np.array(label)
            label = label[np.newaxis, :]
        else:
            label=self.cache_label[index]

        return image, label, name,index

