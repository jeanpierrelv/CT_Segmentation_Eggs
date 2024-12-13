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

    # def load_dicom_folder(folder_path):
    #     #check the order of the files
    #     dicom_stack_tensor_list = []
    #     dicom_folders = sorted([file for file in os.listdir(folder_path)])
    #     for dicom_folder in dicom_folders:
    #         dicom_stack_images = []
    #         file_path = os.path.join(folder_path, dicom_folder)
    #         dicom_files = sorted([file for file in os.listdir(file_path) if file.endswith('.dcm')],
    #                              key=lambda s: int(re.search('\d+', s).group()))
    #         # Filtering the slices with nothing
    #         dicom_files_limit = dicom_files[120:200]
    #         #
    #         for file_name in dicom_files_limit:
    #             # file_path = os.path.join(folder_path, file_name)
    #             dicom_data = pydicom.read_file(file_path + '/' + file_name)
    #             buffer = io.BytesIO()
    #             data = buffer.write(dicom_data[0x7fe0, 0x0010].value)
    #             height = int(np.sqrt(data/2))
    #             dicom_image = np.frombuffer(buffer.getvalue(), dtype=np.uint16).reshape(height,height)  # Convert to float32
    #             dicom_stack_images.append(dicom_image)
    #         dicom_stack_array = np.array(dicom_stack_images)
    #         del(dicom_stack_images, dicom_image)
    #         dicom_stack_array = dicom_stack_array.astype(int)
    #         tensor_3D = torch.from_numpy(dicom_stack_array)
    #         del(dicom_stack_array)
    #         # torch.cat(dicom_stack_tensor, out=tensor_3D)
    #         dicom_stack_tensor_list.append(tensor_3D)
    #         del(tensor_3D)
    #         dicom_stack_tensor = torch.stack(dicom_stack_tensor_list)
    #     del(dicom_stack_tensor_list)    
    #     return dicom_stack_tensor

    # def read_image_CT(self):
    #     if self.type_file == 'dicom':
    #         images = load_dicom_folder(self.img_dir)
    #         # img = pydicom.read_file(self.img_dir)
    #         # data = img.pixel_array.astype(np.float32)
    #     elif self.type_file == 'nii':
    #         img = nib.load(self.img_dir)
    #         data = img.get_fdata()
    #     else:
    #         pass
    #     return data
###############################################################################
    # def read_image_CT(img_dir, aux):
    #         file_path = img_dir+'/'+aux
    #         dicom_files = sorted([file for file in os.listdir(file_path) if file.endswith('.dcm')],
    #                              key=lambda s: int(re.search('\d+', s).group()))
    #         # Filtering the slices with nothing
    #         dicom_files_limit = dicom_files[120:200]
    #         #
    #         dicom_stack_images = []
    #         for file_name in dicom_files_limit:
    #             # file_path = os.path.join(folder_path, file_name)
    #             dicom_data = pydicom.read_file(file_path + '/' + file_name)
    #             buffer = io.BytesIO()
    #             data = buffer.write(dicom_data[0x7fe0, 0x0010].value)
    #             height = int(np.sqrt(data/2))
    #             dicom_image = np.frombuffer(buffer.getvalue(), dtype=np.uint16).reshape(height,height)  # Convert to float32
    #             dicom_stack_images.append(dicom_image)
    #         dicom_stack_array = np.array(dicom_stack_images)
    #         del(dicom_stack_images, dicom_image)
    #         dicom_stack_array = dicom_stack_array.astype(int)
    #         img = torch.from_numpy(dicom_stack_array)
    #         del(dicom_stack_array)
    #         return img
        
    # def read_mask_3D(mask_dir, aux):
    #         file_path = mask_dir+'/'+aux
    #         dicom_files = sorted([file for file in os.listdir(file_path)],
    #                              key=lambda s: int(re.search('\d+', s).group()))
    #         # Filtering the slices with nothing
    #         dicom_files_limit = dicom_files[120:200]
    #         #
    #         data_stack = []
    #         for file_name in dicom_files_limit:
    #             # file_path = os.path.join(folder_path, file_name)
    #             data = np.load(file_path + '/' + file_name)
    #             data_stack.append(data)
    #         data_stack_array = np.array(data_stack)
    #         del(data_stack, data)
    #         data_stack_array = data_stack_array.astype(int)
    #         mask = torch.from_numpy(data_stack_array)
    #         del(data_stack_array)
    #         return mask
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
            img, _ , _= read_image_CT(self.img_dir, aux)
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
            img, _ , _= read_image_CT(self.img_dir, aux)                
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
            return self.crop(img), self.crop(mask)
        else:
            raise Exception("self.crop não está definido.")
        
    def get_loader(self,batch_size,shuffle):
        """
            Retorna um dataloader para o conjunto.
        """
        return DataLoader(self,batch_size=batch_size,shuffle=shuffle)