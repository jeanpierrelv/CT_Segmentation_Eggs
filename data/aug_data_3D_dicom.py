from os import listdir, mkdir
import skimage as ski
import numpy as np
import pydicom
import io
import os
import re
import skimage as ski

#Carrega pastas com imagens
img_dir = input("Diretório com as imagens: ")
mask_dir = input("Diretório com as máscaras de segmentação: ")

#Lista das imagens (as imagens em img_dir e maxk_dir precisam ter o mesmo nome)
img_list = listdir(img_dir)

mkdir('aug')
mkdir('aug/images')
mkdir('aug/masks')

for aux in img_list:
    file_path = img_dir+'/'+aux
    dicom_files = sorted([file for file in os.listdir(file_path) if file.endswith('.dcm')],
                                    key=lambda s: int(re.search('\d+', s).group()))

    dicom_stack_images = []
    for file_name in dicom_files:
        # file_path = os.path.join(folder_path, file_name)
        dicom_data = pydicom.read_file(file_path + '/' + file_name)
        buffer = io.BytesIO()
        data = buffer.write(dicom_data[0x7fe0, 0x0010].value)
        side = int(np.sqrt(data/2))
        dicom_image = np.frombuffer(buffer.getvalue(), dtype=np.uint16).reshape(140,90)#(side,side)  # Convert to float32
        dicom_stack_images.append(dicom_image)