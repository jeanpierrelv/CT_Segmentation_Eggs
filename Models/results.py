import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import numpy as np
from data_prepare import seg_data
from model import UNet

# Define a function to test the model and visualize results
def test_and_visualize(model, test_loader, device):
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():  # Disable gradient computation for inference
        for batch in test_loader:
            images, masks = batch  # Assuming your test_loader provides (images, masks) pairs
            images, masks = images.to(device), masks.to(device)  # Move data to the device

            # Perform inference
            predicted_masks = model(images)

            # Convert tensors to NumPy arrays for visualization
            images = images.cpu().numpy()
            masks = masks.cpu().numpy()
            predicted_masks = predicted_masks.cpu().numpy()

            # Visualize a few examples from the batch
            num_examples = min(images.shape[0], 4)  # Adjust the number of examples to display
            fig, axes = plt.subplots(num_examples, 3, figsize=(12, num_examples * 4))

            for i in range(num_examples=2):
                # Plot the original image
                axes[i, 0].imshow(np.transpose(images[i], (1, 2, 0)), cmap='gray')
                axes[i, 0].set_title('Original Image')
                axes[i, 0].axis('off')

                # Plot the ground truth mask
                axes[i, 1].imshow(masks[i, 0], cmap='viridis')  # Assuming binary masks
                axes[i, 1].set_title('Ground Truth Mask')
                axes[i, 1].axis('off')

                # Plot the predicted mask
                axes[i, 2].imshow(predicted_masks[i, 0], cmap='viridis')  # Assuming binary masks
                axes[i, 2].set_title('Predicted Mask')
                axes[i, 2].axis('off')

            plt.tight_layout()
            plt.show()

# Example usage
# Replace this with your test data and trained model
# test_loader = ...
# trained_model = ...

model_path = input("Diretório do modelo treinado: ")
model_name = input("Nome do modelo.pth: ")
splits_folder = model_path#input('Diretorio com splits: ')
img_folder = model_path + '/images' #input('Diretório com as imagens: ')
mask_folder = model_path + '/masks-binary' #input('Diretório com as máscaras de segmentação: ')
bs = eval(input('Tamanho dos batches: '))

# Load the model from the .pth file
# loaded_model = torch.load(model_path + '/' + model_name)
# trained_model = loaded_model

model = UNet(num_filters=16)  # Replace with your model class and architecture

# Load the trained model weights from the .pth file
checkpoint = torch.load(model_path + '/' + model_name)
model.load_state_dict(checkpoint['model_state_dict'])

n = 0
val_split = np.loadtxt(splits_folder+'/train'+str(n)+'.txt',dtype=str)
val_data = seg_data(val_split,img_folder,mask_folder)
val_data.set_crop
test_loader = val_data.get_loader(bs,False)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Use GPU if available
# trained_model.to(device='cuda')  # Move the model to the device
test_and_visualize(model, test_loader, device)