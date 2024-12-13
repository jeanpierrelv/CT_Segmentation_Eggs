import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import CenterCrop
from torchvision.io import read_image
from torch3D import read_image_CT, read_mask_3D
import numpy as np
import os
from os import listdir
import nibabel as nib
import pydicom
import io
import re
import csv
import pandas as pd

def unpack_mask(mask: torch.Tensor):
    """
        Transforma uma máscara (s, h, w) em um 4-tensor (q,s,h,w):
        h = heigth
        w = width
        s = slices
        q = class 
    """
    s, h, w = mask.size() #Dimensões da imagem
    q = torch.max(mask) #Número de classes
    new_mask = torch.zeros((q+1,h,w))

    #Itera sobre pixels da imagem
    for i in range(h):
        for j in range(w):
            for k in range (s):
                s = int(mask[i][j][k].item())
                new_mask[s][k][i][j] = 1.
    return new_mask

class seg_data(Dataset):
    """
        Classe para preparação de um dataset para a tarefa de segmentação.
    """
    
    def __init__(self,img_list,img_dir,mask_dir):
        """
            Inicializa um dataset a partir de suas instâncias (x) e resultados
            esperados (targets, y).
            img_list: lista com as imagens a serem utilizadas
            img_dir: diretório com as instâncias
            mask_dir: diretório com as máscaras de segmentação

            Nota: é preciso que os arquivos em x_dir e y_dir tenham os mesmos
            nomes. Isto é, um par (x,y) é identificado pelo mesmo nome, mas em
            diretórios distintos.
        """
        super(seg_data,self).__init__()
        self.img_list = img_list
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        # self.type_file = type_file
        #Propriedades
        self.crop = None
        self.crop_dims = None
        self.mean = None
        self.std = None
        
    def __len__(self):
        return len(self.img_list)

    
##############################################################################
    def set_crop(self):
        """
            Define o par de menores dimensões para realizar a operação
            de CenterCrop.
        """
        min_h = 100000000
        min_w = 100000000
        for aux in self.img_list:
            # img = read_image(self.img_dir+'/'+aux)
            img, _, _ = read_image_CT(self.img_dir, aux)
            d, h, w = img.size()
            if min_h > h:
                min_h = h
            if min_w > w:
                min_w = w
        self.crop = CenterCrop((min_h,min_w))
        self.crop_dims = (min_h,min_w)

    def __getitem__(self,idx):
        if self.crop:
            aux = self.img_list[idx]
            # img = read_image(self.img_dir+'/'+aux)
            img, _, _ = read_image_CT(self.img_dir, aux)                
            if self.mean is not None and self.std is not None:
                img = (img.float()-img.min())/(img.max()-img.min())
            # Added a normalize step in real CT images of range between 0-1600 to 0-254
            img = 254 * (img / 1600)
            #--------------------------------------------------------------#
            img = img.float()
            img = img.unsqueeze(0) # Adding an extra dimension for 3D image
            aux = aux.split('.')[0]
            # mask = torch.load(self.mask_dir+'/'+aux+'.pt').long()
            mask = read_mask_3D(self.mask_dir, aux)
            # mask = mask.unsqueeze(0)
            mask = mask.float()
            #mask = unpack_mask(mask).float()
            # Reading the morphometric measurements
            split_aux = int(aux.split('_')[-1])
            # data = pd.read_csv(self.mask_dir + '/' + 'measures.csv')
            # data = data.values
            with open(self.mask_dir + '/' + 'measures.csv', newline='') as csvfile:
                measures = csv.reader(csvfile, delimiter=',')
            # convertedData =[[column.replace(',','.') for column in row] for row in measures]
                row1 = []
                for i, row0 in enumerate(measures):
                    if i == split_aux-1:
                        for k in row0:
                            # row1.append((float(k)-np.max(data))/(np.max(data) - np.min(data)))
                            row1.append(float(k))
            row2 = np.transpose(np.array(row1))#[np.newaxis, :]
            measures_tensor = torch.tensor(row2)    
            # measures_tensor = measures_tensor#.unsqueeze(0)
            # outs = (self.crop(mask), measures_tensor)
            return self.crop(img), self.crop(mask), measures_tensor#outs
        else:
            raise Exception("self.crop não está definido.")
        
    def get_loader(self,batch_size,shuffle):
        """
            Retorna um dataloader para o conjunto.
        """
        return DataLoader(self,batch_size=batch_size,shuffle=shuffle)