import torch
from torch3D import read_image_CT, read_mask_3D
import os
import numpy as np

path_slices = 'data/egg_simulate_v1/slices-crop'
folders = os.listdir(path_slices)
l = []
for i in folders:
    img = read_image_CT(path_slices, i)
    if img.max() == 0:
        l.append(i)
    else:
        pass
   
np.savetxt('files_blank'+'.dat',l,fmt='%s')
