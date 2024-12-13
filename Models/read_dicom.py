import os
import pydicom
import zipfile
import numpy as np
# import pydicom.uid

# Unzip the zip file with the images and masks

dataset_path = './Datasets/'
dataset_name = 'v11.zip'
path_to_zip_file = dataset_path + dataset_name
with zipfile.ZipFile(path_to_zip_file, 'r') as zip_ref:
    zip_ref.extractall(dataset_path)


def load_dicom_folder(folder_path):
    #check the order of the files
    dicom_files = sorted([file for file in os.listdir(folder_path) if file.endswith('.dcm')])
    dicom_images = []
    
    for file_name in dicom_files:
        file_path = os.path.join(folder_path, file_name)
        dicom_data = pydicom.read_file(file_path)
        # dicom_data = pydicom.read_file(file_path, force=True)
        # dicom_data.file_meta.TransferSyntaxUID = pydicom.uid.ImplicitVRLittleEndian  # or whatever is the correct transfer syntax for the file
        # dicom_image = dicom_data.pixel_array
        dicom_image = dicom_data.pixel_array.astype(np.float32)  # Convert to float32
        dicom_images.append(dicom_image)
    
    return dicom_images


# # Provide the path to your DICOM images folder
images_folder_path = dataset_path + 'images/'
masks_folder_path = dataset_path + 'masks/'

foldernames = os.listdir(images_folder_path)
foldernames.sort()

image_folder_list = []
masks_folder_list = []

image_list = []
masks_list = []

dicom_images_samples = []
dicom_masks_samples = []


for foldername in foldernames:
    # List and append the folders paths of images dicom
    image_folder_list.append(images_folder_path + foldername)
    masks_folder_list.append(masks_folder_path + foldername)
    
    # Load DICOM images from each folder
    
    dicom_images = load_dicom_folder(images_folder_path + foldername)
    print(f"Loaded {len(dicom_images)} DICOM images.")
    
    # Load DICOM masks from each folder (if you have masks in a separate folder)
    
    dicom_masks = load_dicom_folder(masks_folder_path + foldername)
    print(f"Loaded {len(dicom_masks)} DICOM masks.")
    
    # Put images stacks in list
    
    dicom_images_samples.append(dicom_images)
    dicom_masks_samples.append(dicom_masks)