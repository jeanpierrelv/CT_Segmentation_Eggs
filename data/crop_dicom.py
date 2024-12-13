import os
import io
import re
import pydicom
import numpy as np

# Function to crop and save DICOM slices
def crop_and_save_dicom(input_folder, output_folder, start_slice, end_slice, vertical_start, vertical_end, horizontal_start, horizontal_end):#, output_prefix="cropped_slice"):
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # List all DICOM files in the input folder
    # dicom_files = [f for f in os.listdir(input_folder) if f.endswith(".dcm")]

    # Sort the DICOM files to ensure correct ordering
    # dicom_files.sort()
    dicom_files = sorted([file for file in os.listdir(input_folder) if file.endswith('.dcm')],
                                 key=lambda s: int(re.search('\d+', s).group()))

    for i, dicom_file in enumerate(dicom_files[start_slice:end_slice], start_slice):
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
        
        # Crop the DICOM slice (modify cropping dimensions as needed)
        cropped_image = dicom_image[vertical_start:vertical_end, horizontal_start:horizontal_end]#.pixel_array[100:400, 100:400]

        # Update DICOM header information (if necessary)
        # Modify fields like ImagePositionPatient, ImageOrientationPatient, etc.

        # Save the cropped DICOM slice
        dicom_data.PixelData = cropped_image.tobytes()
        dicom_data.Rows, dicom_data.Columns = cropped_image.shape
        dicom_data.save_as(output_path)

        print(f"Saved {output_path}")
        
def crop_and_save_masks(input_folder, output_folder, start_slice, end_slice, vertical_start, vertical_end, horizontal_start, horizontal_end):#, output_prefix="cropped_slice"):
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        
    dicom_files = sorted([file for file in os.listdir(input_folder)],
                                key=lambda s: int(re.search('\d+', s).group()))
        
    for i, dicom_file in enumerate(dicom_files[start_slice:end_slice], start_slice):
        input_path = os.path.join(input_folder, dicom_file)
        output_path = os.path.join(output_folder, dicom_file)#f"{output_prefix}_{i:04d}.dcm")
        
        dicom_data = np.load(input_path)
        cropped_image = dicom_data[vertical_start:vertical_end, horizontal_start:horizontal_end]
        np.save(output_path, cropped_image)
        print(f"Saved {output_path}")
        

# Example usage:
input_folder = "data/egg_simulate_v3/slices"
output_folder = "data/egg_simulate_v3/slices-crop"
# start_slice = 120  # Start slice index (inclusive) # Last parameters = 120 - 200 for all
# end_slice = 200   # End slice index (exclusive)
# vertical_start = 90
# vertical_end = 230
# horizontal_start = 116
# horizontal_end = 206

start_slice = 35  # Start slice index (inclusive) # Last parameters = 120 - 200 for all
end_slice = 115   # End slice index (exclusive)
vertical_start = 10
vertical_end = 150
horizontal_start = 30
horizontal_end = 120

input_folder_masks = "data/egg_simulate_v3/masks"
output_folder_masks = "data/egg_simulate_v3/masks-crop"

folders_list_slices = os.listdir(input_folder)

for aux in folders_list_slices:
    crop_and_save_dicom(input_folder +'/' + aux, output_folder +'/' + aux,
                        start_slice, end_slice, 
                        vertical_start, vertical_end,
                        horizontal_start, horizontal_end)
    
    crop_and_save_masks(input_folder_masks +'/' + aux, output_folder_masks +'/' + aux,
                        start_slice, end_slice, 
                        vertical_start, vertical_end,
                        horizontal_start, horizontal_end)