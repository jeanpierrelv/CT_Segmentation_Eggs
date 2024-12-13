import os
import io
import re
import pydicom
import numpy as np
from torchvision.io import read_image
import torch
from PIL import Image


# Function to crop and save DICOM slices
def crop_and_save_dicom(input_folder, output_folder, start_slice, end_slice, vertical_start, vertical_end, horizontal_start, horizontal_end,y_positions, folder_egg_init):#, output_prefix="cropped_slice"):
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

 ################################################
    files = os.listdir(input_folder)

    def sort_key(file_name):
        # Extract the relevant part containing numbers
        # first = file_name.split('.')[-5]
        second = file_name.split('.')[-10]
        numbers_part = f'{second}'
        #numbers_part = ''.join(filter(str.isdigit, file_name))
        # Convert to integers and use as the sorting key
        return float(numbers_part)
    

    sorted_files = sorted(files, key = sort_key)


    rows_tray = 1
    columns_tray = 6
    
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
            egg_one = dicom_data.copy()
            cropped_image_egg_one = cropped_image[:,y[0]:y[1]]
            ## Adding columns without object to complete the size of 140 x 90.
            ## Before this process the extract image had the size of 140 x 75.
            missed_width = width - (y[1]-y[0])
            missed_width1 = missed_width // 2
            missed_width2 = missed_width - missed_width1
            column_to_add = cropped_image[:,0:1]#cropped_image[:,(y[1]-1):y[1]]
            [columns_to_left.append((column_to_add)) for i in range(0,missed_width1)]
            columns_to_left = np.concatenate((columns_to_left), axis=1)
            [columns_to_right.append((column_to_add)) for i in range(0,missed_width2)]
            columns_to_right = np.concatenate((columns_to_right), axis=1)
            final_cropped = np.concatenate((columns_to_left, cropped_image_egg_one, columns_to_right), axis=1)
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
                    folder_egg = folder_egg + 60
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


def crop_and_save_masks(input_folder, output_folder, start_slice, end_slice, vertical_start, vertical_end, horizontal_start, horizontal_end,y_positions, folder_egg_init):#, output_prefix="cropped_slice"):
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

 
    files = os.listdir(input_folder)

    def sort_key(file_name):
        # Extract the relevant part containing numbers
        # first = file_name.split('.')[-5]
        second = file_name.split('.')[-10]
        numbers_part = f'{second}'
        #numbers_part = ''.join(filter(str.isdigit, file_name))
        # Convert to integers and use as the sorting key
        return float(numbers_part)
    

    sorted_files = sorted(files, key = sort_key)


    rows_tray = 1
    columns_tray = 6
    
    aux_i = np.zeros(columns_tray)
    j = 0
    # start = np.zeros((rows_tray,columns_tray))
    start = np.zeros((columns_tray))
    # end = np.zeros((rows_tray,columns_tray))
    end = np.zeros((columns_tray))
    space = np.zeros(columns_tray)# 0, 0, 0, 0, 0
    prev_space = np.zeros(columns_tray)# 0, 0, 0, 0, 0
    
##---------------------------------------------------------------------------##
    min_depth = 40
    if len(sorted_files) < min_depth:
        miss_slices = min_depth - len(sorted_files)
        miss_slices_front = miss_slices // 2
        miss_slices_back = miss_slices - miss_slices_front
        dicom_file = sorted_files[0]

        
        
        input_path = os.path.join(input_folder, dicom_file)
        dicom_data = read_image(input_path)
        cropped_image = dicom_data#[vertical_start:vertical_end, :]
        # Save the cropped DICOM slice


        cropped_image = np.zeros((np.shape(cropped_image)), dtype='uint16')
        cropped_image = np.transpose(cropped_image, (1, 2, 0))
        cropped_image = Image.fromarray(cropped_image.astype('uint16'), 'RGB')
        
        for i in range(1, miss_slices_front+1):
            name_file_front = dicom_file.split('.')
            file_ind_front = int(name_file_front[-10]) - i
            name_file_front[-10] = "{:04d}".format(file_ind_front)
            new_name_front = '.'.join(name_file_front)
            path_save_front = os.path.join(input_folder, new_name_front)
            cropped_image.save(path_save_front, format='PNG')
        
        for j in range(1, miss_slices_back+1):
            name_file_back = sorted_files[-1].split('.')
            file_ind_back = int(name_file_back[-10]) + j
            name_file_back[-10] = "{:04d}".format(file_ind_back)
            new_name_back = '.'.join(name_file_back)
            path_save_back = os.path.join(input_folder, new_name_back)
            # cropped_image = np.transpose(cropped_image, (1, 2, 0))
            # cropped_image = Image.fromarray(cropped_image.astype('uint16'), 'RGB')
            cropped_image.save(path_save_back, format='PNG')

    files = os.listdir(input_folder)    
    sorted_files = sorted(files, key = sort_key)
##---------------------------------------------------------------------------##


    
    # for i, dicom_file in enumerate(dicom_files[start_slice:end_slice], start_slice):
    #     input_path = os.path.join(input_folder, dicom_file)
    #     output_path = os.path.join(output_folder, dicom_file)#f"{output_prefix}_{i:04d}.dcm")
    end_slice = len(sorted_files)    
    for i, dicom_file in enumerate(sorted_files[start_slice:end_slice], start_slice):
        input_path = os.path.join(input_folder, dicom_file)
        output_path = os.path.join(output_folder, dicom_file)    
        
        
        # dicom_data = np.load(input_path)
        dicom_data = read_image(input_path)
        cropped_image = dicom_data[:,vertical_start:vertical_end, :]#horizontal_start:horizontal_end]
        
        # np.save(output_path, cropped_image)
        # print(f"Saved {output_path}")

        # b1 = [(0,0,0)]
        # for i in np.linspace(len(cropped_image[0])/20, len(cropped_image[0])-(len(cropped_image[0])/6), 15, dtype=(int)):
        #     for j in range(len(cropped_image[0][0])):
        #         if tuple(np.array(cropped_image[:,i,j])) in b1:
        #             pass
        #         else:
        #             b1.append(tuple(np.array(cropped_image[:,i,j])))
        #     if len(b1) >= 5:
        #         break
            
        # class_map = {}
        # ind = 0
        # for k in b1:
        #     class_map.update({k:ind})
        #     ind += 1

        class_map = {(0, 0, 0): 0,
         (51, 221, 255): 1,
         (250, 250, 55): 2,
         (61, 245, 61): 3,
         (219, 21, 228): 4}

        mask_aux = torch.zeros(cropped_image.shape[1:], dtype=torch.int)
        for class_rgb, class_label in class_map.items():
            class_indices = torch.all(cropped_image == torch.tensor(class_rgb).view(3, 1, 1), dim=0)
            mask_aux[class_indices] = class_label
        
        cropped_image = mask_aux
        
        width = 90
        for idx, y in enumerate(y_positions[0:]):
            columns_to_right = []
            columns_to_left = []
            # egg_one = cropped_image.copy()
            cropped_image_egg_one = cropped_image[:,y[0]:y[1]]
            missed_width = width - (y[1]-y[0])
            missed_width1 = missed_width // 2
            missed_width2 = missed_width - missed_width1
            column_to_add = cropped_image[:,0:1]#cropped_image[:,(y[1]-1):y[1]]
            [columns_to_left.append((column_to_add)) for i in range(0,missed_width1)]
            columns_to_left = np.concatenate((columns_to_left), axis=1)
            [columns_to_right.append((column_to_add)) for i in range(0,missed_width2)]
            columns_to_right = np.concatenate((columns_to_right), axis=1)
            final_cropped = np.concatenate((columns_to_left, cropped_image_egg_one, columns_to_right), axis=1)
            ##---------------------------------------------------------------##
            cropped_image_egg_one = final_cropped      
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
                    folder_egg = folder_egg + 60
                    dicom_folder = f'egg_{folder_egg}'
                    path_egg_position = os.path.join(output_folder,dicom_folder)
                    if not os.path.exists(path_egg_position):
                        os.makedirs(path_egg_position)
                    dicom_file_new = '.'.join([dicom_file.split('.')[-10], dicom_file.split('.')[-1]])
                    output_egg_path = os.path.join(path_egg_position, dicom_file_new)
                    # aux_filename = dicom_file.split('.')[:-1]
                    # dicom_file = '.'.join(aux_filename)
                    # output_egg_path = os.path.join(path_egg_position, dicom_file)
                    # egg_one.save_as(output_egg_path)
                    np.save(output_egg_path, cropped_image_egg_one)
                    print(f"Saved {output_egg_path}")
            # start[idx] = 0
            # end[idx] = 0
            # y0 = y
            prev_space[idx] = space[idx]
        

        "We need how identificate the depth rows of trays"



#.#-------------------------------------------------------------------------#.# 


# Example usage:
# input_folder = "data/data_backup/real_data_dicom/frescos_bandeja1/images"
# output_folder = "data/data_backup/real_data_dicom/frescos_bandeja1/images-crop"

input_folder = "data/data_backup/real_data_dicom/14dias_bandeja1/images"
output_folder = "data/data_backup/real_data_dicom/14dias_bandeja1/images-crop"


start_slice = 0  # Start slice index (inclusive) # Last parameters = 120 - 200 for all
end_slice = -1   # End slice index (exclusive)
# Positions for (Tray1 - Fresh eggs)
# vertical_start = 150
# vertical_end = 290
# horizontal_start = 0
# horizontal_end = -1
# y_positions = [[24, 99],
#                [100, 175],
#                [178, 253],
#                [252, 327],
#                [328, 403],
#                [402, 477]]

# Positions for (Tray1 - 7 days eggs)
vertical_start = 175
vertical_end = 315
horizontal_start = 0
horizontal_end = -1
y_positions = [[30, 104],
               [102, 178],
               [178, 251],#cortei 1 a mais de 177 -> 178
               [252, 325],#cortei 1 a mais de 251 -> 252
               [327, 400],#cortei 1 a mais de 326 -> 327
               [400, 473]]#cortei 1 a mais de 399 -> 400
# 88, 88, 88, 88, 88 width of crop images
# folders_list_slices = os.listdir(input_folder)


# input_folder_masks = "data/data_backup/real_data_dicom/frescos_bandeja1/masks"
# output_folder_masks = "data/data_backup/real_data_dicom/frescos_bandeja1/masks-crop"

input_folder_masks = "data/data_backup/real_data_dicom/14dias_bandeja1/masks"
output_folder_masks = "data/data_backup/real_data_dicom/14dias_bandeja1/masks-crop"

folders_list_slices = sorted([item for item in os.listdir(input_folder)])
y_aux = y_positions.copy()
folder_egg_init = 0
for idx, aux in enumerate(folders_list_slices):
    if idx==0:
        for i in range(0,2):
            y_aux[i] = [x - 1 for x in y_aux[i]]
        if i ==2:
            y_aux[i][0] = y_aux[i][0]-1
            y_aux[i] = [x - 1 for x in y_aux[i]]
        for i in range(3,4):
            y_aux[i] = [x - 1 for x in y_aux[i]]
        for i in range(4,6):
            y_aux[i] = [x - 2 for x in y_aux[i]]
    if idx==1:
        for i in range(0,6):
            if i ==0:
                y_aux[i] = [x - 2 for x in y_aux[i]]
            elif i ==1:
                y_aux[i] = [x for x in y_aux[i]]
            elif i ==3:
                y_aux[i] = [x-1 for x in y_aux[i]]
            elif i ==4:
                # y_aux[i][1] = y_aux[i][1]-1
                y_aux[i] = [x + 2 for x in y_aux[i]]
            elif i ==5:
                y_aux[i] = [x + 3 for x in y_aux[i]]
    if idx==2:
        for i in range(0,6):
            if i ==1:
                y_aux[i] = [x-1 for x in y_aux[i]]
    if idx==3:
        for i in range(0,6):
            if i ==0:
                y_aux[i] = [x - 1 for x in y_aux[i]]
            elif i ==1:
                y_aux[i][0] = y_aux[i][0]+1
                y_aux[i][1] = y_aux[i][1]-2
            elif i ==2:
                y_aux[i] = [x - 1 for x in y_aux[i]]
            elif i ==3:
                y_aux[i] = [x + 3 for x in y_aux[i]]
            elif i ==4:
                y_aux[i][1] = y_aux[i][1]+2
                y_aux[i] = [x for x in y_aux[i]]
            elif i ==5:
                y_aux[i][0] = y_aux[i][0]+2
                y_aux[i] = [x for x in y_aux[i]]

    if idx==4:
        for i in range(0,6):
            if i == 0:
                y_aux[i] = [x - 2 for x in y_aux[i]]
            elif i ==1:
                y_aux[i] = [x - 1 for x in y_aux[i]]
            elif i ==3:
                y_aux[i] = [x + 3 for x in y_aux[i]]
            elif i ==4:
                y_aux[i] = [x + 2 for x in y_aux[i]]
            elif i ==5:
                y_aux[i][0] = y_aux[i][0]-2
                y_aux[i] = [x + 4 for x in y_aux[i]]
                
    
    crop_and_save_dicom(input_folder +'/' + aux, output_folder, #+'/' + aux, # Add if need to separate by folders like input_folder
                        start_slice, end_slice, 
                        vertical_start, vertical_end,
                        horizontal_start, horizontal_end, y_aux, folder_egg_init)
   
    
    crop_and_save_masks(input_folder_masks +'/' + aux, output_folder_masks, # +'/' + aux,
                        start_slice, end_slice, 
                        vertical_start, vertical_end,
                        horizontal_start, horizontal_end, y_aux, folder_egg_init)
    
    folder_egg_init += 6