import numpy as np
import os
from os import listdir
import nibabel as nib
import pydicom
import io
import re
import torch
from masks_to_bboxes import mask_to_boxes

def read_image_CT(img_dir, aux):
            file_path = img_dir+'/'+aux
            dicom_files = sorted([file for file in os.listdir(file_path) if file.endswith('.dcm') or file.endswith('.IMA')],
                                 key=lambda s: int(re.search('\d+', s).group()))
            # Filtering the slices with nothing
            #dicom_files_limit = dicom_files[0:319] # 120:200 range of image # normal size [0:319]
            #
            dicom_stack_images = []
            for file_name in dicom_files:
                # file_path = os.path.join(folder_path, file_name)
                dicom_data = pydicom.read_file(file_path + '/' + file_name)
                buffer = io.BytesIO()
                data = buffer.write(dicom_data[0x7fe0, 0x0010].value)
                side = int(np.sqrt(data/2))
                dicom_image = np.frombuffer(buffer.getvalue(), dtype=np.uint16).reshape(140,90)#(side,side)  # Convert to float32
                dicom_stack_images.append(dicom_image)
            # pixel_size = 114 #dicom_data.ReconstructionDiameter
            # The next two lines work if you are working with real images, not simulated ones.
            slice_thickness = dicom_data.SliceThickness # depth
            pixel_spacing = dicom_data.PixelSpacing[0] #length
            #########
            dicom_stack_array = np.array(dicom_stack_images)
            del(dicom_stack_images, dicom_image)
            dicom_stack_array = dicom_stack_array.astype(int)
            img = torch.from_numpy(dicom_stack_array)
            del(dicom_stack_array)
            return img, slice_thickness, pixel_spacing
        
def read_mask_3D(mask_dir, aux):
        file_path = mask_dir+'/'+aux
        dicom_files = sorted([file for file in os.listdir(file_path)],
                                key=lambda s: int(re.search('\d+', s).group()))
        # Filtering the slices with nothing
        #dicom_files_limit = dicom_files[120:200]
        #
        data_stack = []
        for file_name in dicom_files:
            # file_path = os.path.join(folder_path, file_name)
            data = np.load(file_path + '/' + file_name)
            data_stack.append(data)
        data_stack_array = np.array(data_stack)
        del(data_stack, data)
        data_stack_array = data_stack_array.astype(int)
        mask = torch.from_numpy(data_stack_array)
        del(data_stack_array)
        return mask

def volume_mask_3D(mask):#(img_dir, mask_dir, aux):
        """
        This function count the number of voxels estimated per class as a previous
        count to multiply by the volume of a voxel in another function.
        """
        voxel_true = (mask==0)
        voxel_sum_background = voxel_true.sum()
        voxel_true = (mask==1)
        voxel_sum_shell = voxel_true.sum()
        voxel_true = (mask==2)
        voxel_sum_yolk = voxel_true.sum()
        voxel_true = (mask==3)
        voxel_sum_albumen = voxel_true.sum()
        voxel_true = (mask==4)
        voxel_sum_air = voxel_true.sum()

        volumes = [voxel_sum_shell, voxel_sum_yolk, voxel_sum_albumen, voxel_sum_air, voxel_sum_background]
        return volumes

def volume_mask_3D_detail(masks, img_dir, img_list):#(img_dir, mask_dir, aux):
        """
        This function count the number of voxels estimated per class as a previous
        count to multiply by the volume of a voxel in another function.
        """
        volumes_one_stack = []
        dimensions_one_stack = []
        i=0
        for mask_one_egg in masks:
            aux = img_list[i]
            _, depth, length= read_image_CT(img_dir, aux)
            
            voxel_true = (mask_one_egg==0)
            voxel_sum_background = voxel_true.sum()
            voxel_true = (mask_one_egg==1)
            voxel_sum_shell = voxel_true.sum()
            voxel_true = (mask_one_egg==2)
            voxel_sum_yolk = voxel_true.sum()
            voxel_true = (mask_one_egg==3)
            voxel_sum_albumen = voxel_true.sum()
            voxel_true = (mask_one_egg==4)
            voxel_sum_air = voxel_true.sum()
            
            volumes_one = [voxel_sum_shell, voxel_sum_yolk, voxel_sum_albumen, voxel_sum_air, voxel_sum_background]
            volumes_one_stack.append(volumes_one)
            dimensions_one = [depth, length]
            dimensions_one_stack.append(dimensions_one)
            i += 1
        volumes = np.sum(volumes_one_stack, axis=0)
        # volumes = [voxel_sum_shell, voxel_sum_yolk, voxel_sum_albumen, voxel_sum_air, voxel_sum_background]
        return volumes, volumes_one_stack, dimensions_one_stack
    
def measurements_3D(masks):
    """
    This function have the objective of obtain the height, width and thickness of eggs(masks)
    in terms of numbers of Voxels
    """
    coordinates = mask_to_boxes(masks)
    depth1 = coordinates[1]["boxes"][:,0,2]
    depth2 = coordinates[1]["boxes"][:,0,5]
    # subtract 5 because of missing points in the middle. This is only for the simulator
    mid_point = (depth2 - ((depth2-depth1)/2))
    thickness_shells =[]
    heights =[]
    widths = []
    for i in range(0,len(mid_point)):
       pos = np.where(masks[i][int(np.array(mid_point)[i])])
       xmin = np.min(pos[1])
       xmax = np.max(pos[1])
       ymin = np.min(pos[0])
       ymax = np.max(pos[0])
       vertical_axis = xmax-((xmax-xmin)/2)
       horizontal_axis = ymax-((ymax-ymin)/2)-5
       vertical_shell = sum(masks[i][int(np.array(mid_point)[i])][:, int(vertical_axis)]==1)
       horizontal_shell = sum(masks[i][int(np.array(mid_point)[i])][int(horizontal_axis),:]==1)
       height = sum(masks[i][int(np.array(mid_point)[i])][:, int(vertical_axis)]!=0)
       width = sum(masks[i][int(np.array(mid_point)[i])][int(horizontal_axis),:]!=0)
       thickness_shell = (horizontal_shell + vertical_shell)/4
       thickness_shells.append(thickness_shell)
       heights.append(height)
       widths.append(width)
    return [thickness_shells, heights, widths]