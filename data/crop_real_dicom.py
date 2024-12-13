import os
import io
import re
import pydicom
import numpy as np

# Function to crop and save DICOM slices
def crop_and_save_dicom(input_folder, output_folder, start_slice, end_slice, vertical_start, vertical_end, horizontal_start, horizontal_end,y_positions):#, output_prefix="cropped_slice"):
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # List all DICOM files in the input folder
    # dicom_files = [f for f in os.listdir(input_folder) if f.endswith(".dcm")]

    # Sort the DICOM files to ensure correct ordering
    # dicom_files.sort()
    # dicom_files = sorted([file for file in os.listdir(input_folder) if file.endswith('.dcm')])#,
                                 #key=lambda s: int(re.search('\d+', s).group()))
 ################################################
    dicom_files = os.listdir(input_folder)

    def sort_key(file_name):
        # Extract the relevant part containing numbers
        # first = file_name.split('.')[-5]
        second = file_name.split('.')[-4]
        numbers_part = f'{second}'
        #numbers_part = ''.join(filter(str.isdigit, file_name))
        # Convert to integers and use as the sorting key
        return float(numbers_part)
    

    sorted_files = sorted(dicom_files, key = sort_key)


    rows_tray = 6
    columns_tray = 5
    
    aux_i = np.zeros(columns_tray)
    j = 0
    # start = np.zeros((rows_tray,columns_tray))
    start = np.zeros((columns_tray))
    # end = np.zeros((rows_tray,columns_tray))
    end = np.zeros((columns_tray))
    space = np.zeros(columns_tray)# 0, 0, 0, 0, 0
    prev_space = np.zeros(columns_tray)# 0, 0, 0, 0, 0
    folder_egg = 0
######################################
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
        cropped_image = dicom_image[vertical_start:vertical_end, horizontal_start:horizontal_end]#.pixel_array[100:400, 100:400]

        # Update DICOM header information (if necessary)
        # Modify fields like ImagePositionPatient, ImageOrientationPatient, etc.

        # Save the cropped DICOM slice
        dicom_data.PixelData = cropped_image.tobytes()
        dicom_data.Rows, dicom_data.Columns = cropped_image.shape
        # dicom_data.save_as(output_path)

        # Individual Crop for each egg (In this case 6 eggs in a row)

        # y1, y2, y3, y4, y5, y6 = y_positions[1:]
        y0 = y_positions[0]
        for idx, y in enumerate(y_positions[1:]):
            
            
            egg_one = dicom_data.copy()
            cropped_image_egg_one = cropped_image[:,y0:y]
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
                    folder_egg = (columns_tray * aux_i[idx]) + (idx + 1)
                    dicom_folder = f'egg_{folder_egg}'
                    path_egg_position = os.path.join(output_folder,dicom_folder)
                    if not os.path.exists(path_egg_position):
                        os.makedirs(path_egg_position)
                    output_egg_path = os.path.join(path_egg_position, dicom_file)
                    egg_one.save_as(output_egg_path)
                    print(f"Saved {output_egg_path}")
            # start[idx] = 0
            # end[idx] = 0
            y0 = y
            prev_space[idx] = space[idx]
        
        # prev_space = space
            ##########
            # if mean_egg_slice > 62000:
            #     # print("not considered")
            #     space[idx] = 0
            #     if space[idx] == 0 and prev_space[idx] == 1:
            #         end[idx] = 1
            #         start[idx] = 0
            #         aux_i[idx] += 1
                    
            # else:
            #     # print("considered")
            #     dicom_folder = f'egg_{folder_egg}'
            #     path_egg_position = os.path.join(output_folder,dicom_folder)
            #     if not os.path.exists(path_egg_position):
            #         os.makedirs(path_egg_position)
            #     space[idx] = 1
            #     if space[idx] == 1 and prev_space[idx] == 0:
            #         start[idx] = 1
            #         end[idx] = 0
            #     # elif start[idx] == 1 and end[idx] == 0:
            #         "append the slice"
            #         output_egg_path = os.path.join(path_egg_position, dicom_file)
            #         egg_one.save_as(output_egg_path)
            #         print(f"Saved {output_egg_path}")
        
            # y0 = y
        # prev_space = space
        "We need how identificate the depth rows of trays"
        

# Example usage:
input_folder = "data/ovo_first_data/test_data_egg"
output_folder = "data/ovo_first_data/test_data_egg-crop"


start_slice = 0  # Start slice index (inclusive) # Last parameters = 120 - 200 for all
end_slice = -1   # End slice index (exclusive)
vertical_start = 160
vertical_end = 290
horizontal_start = 0
horizontal_end = -1
y_positions = 62, 149, 237, 325, 413, 501
# 88, 88, 88, 88, 88 width of crop images
folders_list_slices = os.listdir(input_folder)

input_folder_masks = "data/ovo_first_data/masks"
output_folder_masks = "data/egg_simulate_v3/masks-crop"

folders_list_slices = os.listdir(input_folder)

for aux in folders_list_slices:
    crop_and_save_dicom(input_folder +'/' + aux, output_folder +'/' + aux,
                        start_slice, end_slice, 
                        vertical_start, vertical_end,
                        horizontal_start, horizontal_end, y_positions)
    
    # crop_and_save_masks(input_folder_masks +'/' + aux, output_folder_masks +'/' + aux,
    #                     start_slice, end_slice, 
    #                     vertical_start, vertical_end,
    #                     horizontal_start, horizontal_end)