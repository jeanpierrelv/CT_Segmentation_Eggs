import os
import numpy as np
from PIL import Image


def convert_npy_to_jpeg(input_folder, output_folder):
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # List all .npy files in the input folder
    npy_files = [f for f in os.listdir(input_folder) if f.endswith('.npy')]

    for npy_file in npy_files:
        # Load the .npy file
        npy_path = os.path.join(input_folder, npy_file)
        array = np.load(npy_path)

        # Normalize the array to 0-255 and convert to uint8
        normalized_array = ((array - array.min()) / (array.max() - array.min()) * 255).astype(np.uint8)

        # Convert numpy array to PIL Image
        image = Image.fromarray(normalized_array)

        # Save as JPEG
        jpeg_file = os.path.splitext(npy_file)[0] + '.jpeg'
        jpeg_path = os.path.join(output_folder, jpeg_file)
        image.save(jpeg_path)

        print(f"Saved {npy_file} as {jpeg_file}")


# Example usage
input_root = 'data/egg_simulate_v3/masks-crop'
output_root = 'data/egg_simulate_v3/masks-crop-annoted'

folders_name = [f for f in os.listdir(input_root)]
for folder in folders_name:
    convert_npy_to_jpeg(input_root + '/' + folder, output_root + '/' + folder)