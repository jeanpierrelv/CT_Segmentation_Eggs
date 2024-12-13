#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 11:35:55 2024

@author: jean
"""

import os
import io
from torch3D import read_image_CT, read_mask_3D
import math
import pydicom
import numpy as np
from scipy.ndimage import label
import matplotlib.pyplot as plt

def sort_key(file_name):
    # Extract the relevant part containing numbers
    # first = file_name.split('.')[-5]
    second = file_name.split('.')[-10]
    numbers_part = f'{second}'
    #numbers_part = ''.join(filter(str.isdigit, file_name))
    # Convert to integers and use as the sorting key
    return float(numbers_part)


def find_axis_boxes(root_path_in, root_path_out, min_size):
    
    # def sort_key(file_name):
    #     # Extract the relevant part containing numbers
    #     # first = file_name.split('.')[-5]
    #     second = file_name.split('.')[-10]
    #     numbers_part = f'{second}'
    #     #numbers_part = ''.join(filter(str.isdigit, file_name))
    #     # Convert to integers and use as the sorting key
    #     return float(numbers_part)
    
    # root_path = self.root_path_in
    # root_path_out = self.root_path_out
    list_path_in = os.listdir(root_path_in)
    for make_dir in list_path_in:    
        if not os.path.exists(root_path_out + '/' + make_dir):
                os.makedirs(root_path_out + '/' + make_dir)
           
    tray_folders = os.listdir(root_path_in)
    

    for tray in tray_folders:
        root_tray = root_path_in + '/' + tray
        sorted_root_tray = sorted(os.listdir(root_tray))
        for row in sorted_root_tray:
            root_row = root_tray + '/' + row
            number_middle_slice = math.ceil(len(os.listdir(root_row))/2)
            files = os.listdir(root_row)
            sorted_files = sorted(files, key = sort_key)
            # Adding the 2 middle slices to estimate the x-axis points for the bounding boxes
            # number_middle_slices = [number_middle_slice, number_middle_slice + 1]
            
            # # Calculating the xmin and xmax per egg
            # bounding_boxes_max = []
            # for number_middle_slice in number_middle_slices:
            name_middle_slice = sorted_files[number_middle_slice]
            root_middle_slice = root_path_in + '/' + tray + '/' + row + '/' + name_middle_slice
            
            dicom_data = pydicom.read_file(root_middle_slice)
            buffer = io.BytesIO()
            data = buffer.write(dicom_data[0x7fe0, 0x0010].value)
            side = int(np.sqrt(data/2))

            dicom_image = np.frombuffer(buffer.getvalue(), dtype=np.uint16).reshape(side,side)
            
            initial_y_crop = round(side/3)
            cropped_image = dicom_image[0:(side-initial_y_crop),:]
            
            threshold=1100
            binary_image = (cropped_image > threshold).astype(int)
            # plt.imshow(binary_image)

            # Label connected components
            labeled_image, num_features = label(binary_image)
            
            # List to store bounding boxes
            bounding_boxes = []
            
            # Loop through each labeled component
            # for i in range(1, num_features + 1):
            for i in range(1, num_features + 1):
                # Get the coordinates of each component
                component = np.where(labeled_image == i)
                
                # Calculate the size of the component
                component_size = len(component[0])  # Number of pixels in the component
                # component_size = ((component[0][1]+1) -component[0][0])*((component[1][1]+1) -component[1][0])
                
                
                # Filter out small components
                if component_size < min_size:
                    continue  # Skip this component if it's too small
                
                y_min, y_max = component[0].min(), component[0].max()
                x_min, x_max = component[1].min(), component[1].max()
                                    
                # Append bounding box
                bounding_boxes.append((x_min, y_min, x_max, y_max))
            # bounding_boxes_max.append(bounding_boxes)
                
            y1 = 300
            y2 = 0
            # fig, ax = plt.subplots(1)
            # ax.imshow(binary_image, cmap='gray')
            for bbox in bounding_boxes:
                x_min, y_min, x_max, y_max = bbox
                
                if y_min < y1:
                    y1 = y_min
                if y_max > y2:
                    y2 = y_max
                # print(y2-y1)
                # print(y1)
                # print(y2)
            #     rect = plt.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, edgecolor='red', facecolor='none')
            #     ax.add_patch(rect)
            # plt.show()
            
            height_missing = 140 - (y2-y1)
            y2_estimated = round(height_missing/2) + y2
            y1_estimated = y1 -(140-(y2_estimated-y1))
            bbox_estimated = []
            
            # fig, ax = plt.subplots(1)
            # ax.imshow(binary_image, cmap='gray')
            for bbox1 in bounding_boxes:
                
                x_min, y_min, x_max, y_max = bbox1
                x_min = x_min - 3
                x_max = x_max + 3
                bbox_estimated.append((x_min, y1_estimated, x_max, y2_estimated))
                
            #     rect = plt.Rectangle((x_min, y1_estimated), x_max - x_min, y2_estimated - y1_estimated, edgecolor='red', facecolor='none')
            #     ax.add_patch(rect)
            # plt.show()
            return bbox_estimated
            




def crop_and_save_dicom(input_folder, output_folder, start_slice, end_slice, vertical_start, vertical_end, horizontal_start, horizontal_end,y_positions, folder_egg_init, columns_tray):#, output_prefix="cropped_slice"):
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

 ################################################
    files = os.listdir(input_folder)

    # def sort_key(file_name):
    #     # Extract the relevant part containing numbers
    #     # first = file_name.split('.')[-5]
    #     second = file_name.split('.')[-10]
    #     numbers_part = f'{second}'
    #     #numbers_part = ''.join(filter(str.isdigit, file_name))
    #     # Convert to integers and use as the sorting key
    #     return float(numbers_part)
    

    sorted_files = sorted(files, key = sort_key)


    rows_tray = 1
    # numbers of columns 
    columns_tray = columns_tray # 6
    
    aux_i = np.zeros(columns_tray)
    j = 0
    # start = np.zeros((rows_tray,columns_tray))
    start = np.zeros((columns_tray))
    # end = np.zeros((rows_tray,columns_tray))
    end = np.zeros((columns_tray))
    space = np.zeros(columns_tray)# 0, 0, 0, 0, 0
    prev_space = np.zeros(columns_tray)# 0, 0, 0, 0, 0
    # folder_egg = 0
    

##---------------------------------------------------------------------------##
    min_depth = 40
    if len(sorted_files) < min_depth:
        miss_slices = min_depth - len(sorted_files)
        miss_slices_front = miss_slices // 2
        miss_slices_back = miss_slices - miss_slices_front
        dicom_file = sorted_files[0]#sorted_files[start_slice:end_slice][0]

        
        
        input_path = os.path.join(input_folder, dicom_file)
        dicom_data = pydicom.read_file(input_path)
        buffer = io.BytesIO()
        data = buffer.write(dicom_data[0x7fe0, 0x0010].value)
        side = int(np.sqrt(data/2))

        dicom_image = np.frombuffer(buffer.getvalue(), dtype=np.uint16).reshape(side,side)
        cropped_image = dicom_image#[vertical_start:vertical_end, :]
        # Save the cropped DICOM slice


        cropped_image = np.zeros((np.shape(cropped_image)), dtype='uint16')
        dicom_data.PixelData = cropped_image.tobytes()
        dicom_data.Rows, dicom_data.Columns = cropped_image.shape
        for i in range(1, miss_slices_front+1):
            name_file_front = dicom_file.split('.')
            file_ind_front = int(name_file_front[-10]) - i
            name_file_front[-10] = "{:04d}".format(file_ind_front)
            new_name_front = '.'.join(name_file_front)
            path_save_front = os.path.join(input_folder, new_name_front)
            dicom_data.save_as(path_save_front)
        
        for j in range(1, miss_slices_back+1):
            name_file_back = sorted_files[-1].split('.')#sorted_files[start_slice:end_slice][-1].split('.')
            file_ind_back = int(name_file_back[-10]) + j
            name_file_back[-10] = "{:04d}".format(file_ind_back)
            new_name_back = '.'.join(name_file_back)
            path_save_back = os.path.join(input_folder, new_name_back)
            dicom_data.save_as(path_save_back)
        
    files = os.listdir(input_folder)    
    sorted_files = sorted(files, key = sort_key)
##---------------------------------------------------------------------------##
######################################
    end_slice = len(sorted_files)   
    for i, dicom_file in enumerate(sorted_files[start_slice:end_slice], start_slice):
        input_path = os.path.join(input_folder, dicom_file)
        output_path = os.path.join(output_folder, dicom_file)#f"{output_prefix}_{i:04d}.dcm")
        
        # Read the DICOM file
        # dicom_data = pydicom.dcmread(input_path)
        dicom_data = pydicom.read_file(input_path)
        buffer = io.BytesIO()
        data = buffer.write(dicom_data[0x7fe0, 0x0010].value)
        side = int(np.sqrt(data/2))

        dicom_image = np.frombuffer(buffer.getvalue(), dtype=np.uint16).reshape(side,side)  # Convert to float32
        #dicom_image = np.array(dicom_image)
        
        # Crop the DICOM slice (modify cropping dimensions as needed): This is the first crop with a rows of eggs
        cropped_image = dicom_image[vertical_start:vertical_end, :]#horizontal_start:horizontal_end]#.pixel_array[100:400, 100:400]

        
        # Update DICOM header information (if necessary)
        # Modify fields like ImagePositionPatient, ImageOrientationPatient, etc.

        # Save the cropped DICOM slice
        dicom_data.PixelData = cropped_image.tobytes()
        dicom_data.Rows, dicom_data.Columns = cropped_image.shape
        # dicom_data.save_as(output_path)


        # Individual Crop for each egg (In this case 6 eggs in a row)

        # y1, y2, y3, y4, y5, y6 = y_positions[1:]
        # y0 = y_positions[0]
        width = 90
        #np.array([], dtype=('uint16'))#torch.tensor([])
        for idx, y in enumerate(y_positions[0:]):
            columns_to_right = []
            columns_to_left = []
            total_columns = []
            egg_one = dicom_data.copy()
            cropped_image_egg_one = cropped_image[:,y[0]:y[1]]
            ## Adding columns without object to complete the size of 140 x 90.
            ## Before this process the extract image had the size of 140 x 75.
            missed_width = width - (y[1]-y[0])
            missed_width1 = missed_width // 2
            missed_width2 = missed_width - missed_width1
            column_to_add = cropped_image[:,0:1]#cropped_image[:,(y[1]-1):y[1]]
            [columns_to_left.append((column_to_add)) for i in range(0,missed_width1)]
            if columns_to_left != []:
                columns_to_left = np.concatenate((columns_to_left), axis=1)
                total_columns.append(columns_to_left)
            total_columns.append(cropped_image_egg_one)
            [columns_to_right.append((column_to_add)) for i in range(0,missed_width2)]
            if columns_to_right != []:
                columns_to_right = np.concatenate((columns_to_right), axis=1)
                total_columns.append(columns_to_right)
                # (columns_to_left, cropped_image_egg_one, columns_to_right)
            final_cropped = np.concatenate(total_columns, axis=1)
            ##---------------------------------------------------------------##
            cropped_image_egg_one = final_cropped
            egg_one.PixelData = cropped_image_egg_one.tobytes()
            egg_one.Rows, egg_one.Columns = cropped_image_egg_one.shape
            
            mean_egg_slice = np.mean(cropped_image_egg_one)
            
            ##########
            if mean_egg_slice > 63500:
                space[idx] = 0
                if space[idx] == 0 and prev_space[idx] == 1:
                    # end[idx] = 1
                    aux_i[idx] += 1
            else:
                space[idx] = 1  
                if space[idx] == 1: #and prev_space[idx] == 0:
                    # start[iaux_idx] = 1
                    folder_egg = int((columns_tray * aux_i[idx]) + (idx + 1)) + folder_egg_init
                    folder_egg = folder_egg #+ 30
                    dicom_folder = f'egg_{folder_egg}'
                    path_egg_position = os.path.join(output_folder,dicom_folder)
                    if not os.path.exists(path_egg_position):
                        os.makedirs(path_egg_position)
                    dicom_file_new = '.'.join([dicom_file.split('.')[-10], dicom_file.split('.')[-1]])
                    output_egg_path = os.path.join(path_egg_position, dicom_file_new)
                    egg_one.save_as(output_egg_path)
                    print(f"Saved {output_egg_path}")
            # start[idx] = 0
            # end[idx] = 0
            # y0 = y
            prev_space[idx] = space[idx]
        

        "We need how identificate the depth rows of trays"
        
        

def axis_to_crop(root_in, root_out):

    root_path_in = root_in#'bandejas'
    root_path_out = root_out#'bandejas1'
    
    min_size = 100
    bbox_estimated = find_axis_boxes(root_path_in, root_path_out, min_size)
    
    # input_folder = root_path_in
    # output_folder = root_path_out
    start_slice = 0  # Start slice index (inclusive) # Last parameters = 120 - 200 for all
    end_slice = -1   # End slice index (exclusive)
    horizontal_start = 0
    horizontal_end = -1
    vertical_start = bbox_estimated[0][1]
    vertical_end = bbox_estimated[0][3]
    x_positions = []
    for x_ax in bbox_estimated:
        x_positions.append([x_ax[0], x_ax[2]])
    
    list_root_path_in = os.listdir(root_path_in)
    columns_tray = len(x_positions)
    for path in list_root_path_in:
        folder_egg_init = 0
        pre_input_folder = root_path_in + '/' + path
        pre_output_folder = root_path_out + '/' + path
        list_sorted_pre_input_folder = sorted(os.listdir(pre_input_folder))
        for aux in list_sorted_pre_input_folder:
            crop_and_save_dicom(pre_input_folder + '/' + aux, pre_output_folder, start_slice, end_slice, vertical_start, vertical_end, horizontal_start, horizontal_end, x_positions, folder_egg_init, columns_tray)#, output_prefix="cropped_slice")
            # number of eggs in a row to add in the next row
            folder_egg_init += columns_tray
    