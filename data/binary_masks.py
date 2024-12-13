import numpy as np
import skimage as ski
from os import listdir, mkdir
import torch

#Carrega máscaras de segmentação
masks_dir = input('Diretório com as máscaras de segmentação: ')
masks_list = listdir(masks_dir)

#Nome do diretório de saída
out_dir = input('Nome do diretório de saída: ')
#Cria diretório de saída
mkdir(out_dir)

for aux in masks_list:
    #Carrega imagem
    img = ski.io.imread(masks_dir+'/'+aux,as_gray=True)
    h = img.shape[0]
    w = img.shape[1]

    #Máscara em forma de uma distribuição de classes
    q_mask = np.zeros((h,w),dtype=np.uint8)

    #Itera sobre os pixels da imagem
    for i in range(h):
        for j in range(w):
            #Identifica cada pixel com uma classe
            if img[i][j] < 0.5:
                q_mask[i][j] = 1
            else:
                q_mask[i][j] = 0
    aux = aux.split('.')[0]
    torch.save(torch.tensor(q_mask,dtype=torch.uint8),out_dir+'/'+aux+'.pt')