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
        for batch in test_loader.dataset:
            images, masks = batch  # Assuming your test_loader provides (images, masks) pairs
            images, masks = images.to(device), masks.to(device)  # Move data to the device

            # Perform inference
            images = images.unsqueeze(0)
            model = model.to(device)
            predicted_masks = model(images)

            # Convert tensors to NumPy arrays for visualization
            images = images.cpu().numpy()
            masks = masks.cpu().numpy()
            predicted_masks = predicted_masks.cpu().numpy()

            # Visualize a few examples from the batch
            num_examples = 1 # Adjust the number of examples to display
            fig, axes = plt.subplots(num_examples, 3, figsize=(12, num_examples * 4))
            
            # Reajust the dimensions of images, ground truth and predicted masks
            # if len(images.shape == 5):
            images = images.squeeze(0)
            images = images.squeeze(1)
            images = images/255
            predicted_masks = (predicted_masks.squeeze(0))
            predicted_masks = predicted_masks[0] + predicted_masks [1]



            # Plot the original image
            axes[0].imshow(np.transpose(images, (1, 2, 0)), cmap='gray')
            axes[0].set_title('Original Image')
            axes[0].axis('off')

            # Plot the ground truth mask
            axes[1].imshow(masks, cmap='viridis')  # Assuming binary masks
            axes[1].set_title('Ground Truth Mask')
            axes[1].axis('off')

            # Plot the predicted mask
            axes[2].imshow(predicted_masks, cmap='viridis')  # Assuming binary masks
            axes[2].set_title('Predicted Mask')
            axes[2].axis('off')

            plt.tight_layout()
            plt.show()

model_path = input("Diretório do modelo treinado: ")
# model_name = input("Nome do modelo.pth: ")
splits_folder = input('Diretorio com splits: ')
img_folder = input('Diretório com as imagens: ')
mask_folder = input('Diretório com as máscaras de segmentação: ')
# bs = eval(input('Tamanho dos batches: '))

model = UNet(num_filters=8)  # Replace with your model class and architecture

# Load the trained model weights from the .pth file
checkpoint = torch.load(model_path)
model.load_state_dict(checkpoint)

n = 0
test_split = np.loadtxt(splits_folder+'/test'+str(n)+'.txt',dtype=str)
test_data = seg_data(test_split,img_folder,mask_folder)
test_data.set_crop()
test_loader = test_data.get_loader(1,False)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Use GPU if available
# trained_model.to(device='cuda')  # Move the model to the device
test_and_visualize(model, test_loader, device)