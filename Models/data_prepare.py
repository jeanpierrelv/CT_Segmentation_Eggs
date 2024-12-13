import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import CenterCrop
from torchvision.io import read_image
import numpy as np
from os import listdir

def unpack_mask(mask: torch.Tensor):
    """
        Transforma uma máscara (h,w) em um 3-tensor (q,h,w), onde q é o número
        de classes. 
    """
    h, w = mask.size() #Dimensões da imagem
    q = torch.max(mask) #Número de classes
    new_mask = torch.zeros((q+1,h,w))

    #Itera sobre pixels da imagem
    for i in range(h):
        for j in range(w):
            s = int(mask[i][j].item())
            new_mask[s][i][j] = 1.
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
        #Propriedades
        self.crop = None
        self.crop_dims = None
        self.mean = None
        self.std = None
        
    def __len__(self):
        return len(self.img_list)

    def set_crop(self):
        """
            Define o par de menores dimensões para realizar a operação
            de CenterCrop.
        """
        min_h = 100000000
        min_w = 100000000
        for aux in self.img_list:
            img = read_image(self.img_dir+'/'+aux)
            c, h, w = img.size()
            if min_h > h:
                min_h = h
            if min_w > w:
                min_w = w
        self.crop = CenterCrop((min_h,min_w))
        self.crop_dims = (min_h,min_w)

    def __getitem__(self,idx):
        if self.crop:
            aux = self.img_list[idx]
            img = read_image(self.img_dir+'/'+aux)
            if self.mean is not None and self.std is not None:
                img = (img.float()-img.min())/(img.max()-img.min())
            img = img.float()
            img = img.unsqueeze(1) # Adding an extra dimension for 3D image
            aux = aux.split('.')[0]
            mask = torch.load(self.mask_dir+'/'+aux+'.pt').long()
            #mask = unpack_mask(mask).float()
            return self.crop(img), self.crop(mask)
        else:
            raise Exception("self.crop não está definido.")
        
    def get_loader(self,batch_size,shuffle):
        """
            Retorna um dataloader para o conjunto.
        """
        return DataLoader(self,batch_size=batch_size,shuffle=shuffle)