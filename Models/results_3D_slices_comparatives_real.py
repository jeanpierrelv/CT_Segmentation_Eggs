import torch
import matplotlib.pyplot as plt
from model_3D import UNet, FCN, MaskRCNN, ConvNeXt
from torch3D import read_image_CT, read_mask_3D
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns


import matplotlib.cm as cm

from matplotlib.colors import Normalize
from matplotlib.colorbar import ColorbarBase
from matplotlib.cm import ScalarMappable


img_dir = 'data/data_backup/real_data_dicom/mix_filter_data/images-crop'
mask_dir = 'data/data_backup/real_data_dicom/mix_filter_data/masks-crop'

# aux = 'egg_54'
aux = 'egg_53'#'gd2k3gottt'#'wws8ll1s5s'#'ss512nmkjq'#testv3 gd2k3gottt #testv2 myj5ix0cvv
# model_path1 = 'data/data_backup/real_data_dicom/mix_filter_data/results_filter_mix/unet/segmentation_only/model_state_0.pth'
# model_path2 = 'data/data_backup/real_data_dicom/mix_filter_data/results_filter_mix/fcn/segmentation_only/model_state_0.pth'
model_path1 = 'data/data_backup/real_data_dicom/mix_filter_data/results_filter_mix_reviewed/unet_segmentation/4to_exper_com_norm_0_0001_epoch_1000_batch_25/unet/model_state_0.pth'
model_path2 = 'data/data_backup/real_data_dicom/mix_filter_data/results_filter_mix_reviewed/fcn_segmentation/1er_exper_com_norm_0_00001_epoch_1000_batch_15/fcn/model_state_0.pth'

## Models trained with simulated images ##
# img_dir = 'data/egg_simulate_v3/slices-crop'
# mask_dir = 'data/egg_simulate_v3/masks-crop'
# aux = 'a5no8uez3s'#'gd2k3gottt'#'wws8ll1s5s'#'ss512nmkjq'#testv3 gd2k3gottt #testv2 myj5ix0cvv
# model_path1 = 'results/unet/split_80_10_10/epochs_150/model_state_0.pth'
# model_path2 = 'results/fcn/split_80_10_10/epochs_1000_lr_00001/model_state_0.pth'
## ------------------------------------ ##

n_classes = 5
slices = [16,18,20,22,24]

img_main, _, _ = read_image_CT(img_dir, aux)
img = img_main.unsqueeze(0) # Adding a Channel dimension and Batch dimension, for have 5D as the input requirement in Conv3D
img = img.unsqueeze(0) # Adding , for have 5D as the input requirement in Conv3D
img = img.float()


mask = read_mask_3D(mask_dir, aux)
model1 = UNet(num_filters=8,n_classes=n_classes)
model2 = FCN(1, 5)
# model = ConvNeXt()
# model = MaskRCNN(5)

checkpoint1 = torch.load(model_path1)
model1.load_state_dict(checkpoint1)

checkpoint2 = torch.load(model_path2)
model2.load_state_dict(checkpoint2)

ground_truth_mask = mask
model1.eval()
model2.eval()
predicted_mask1 = model1(img)
predicted_mask2 = model2(img)

with torch.no_grad():
    predicted_mask1 = predicted_mask1.squeeze(0)
    predicted_mask2 = predicted_mask2.squeeze(0)
    
    outs_arg1 = torch.argmax(predicted_mask1,dim=0)
    outs_arg2 = torch.argmax(predicted_mask2,dim=0)
    predicted_mask_interv1 = predicted_mask1
    predicted_mask_interv2 = predicted_mask2
####################################################################################
    fig, axes = plt.subplots(5, (2*n_classes)+4, figsize=(15, 10), sharex=True, sharey=True)
    plt.subplots_adjust(hspace=0, wspace=-0.5)
    plt.xticks([10,45,80])
    plt.yticks([10,70,130])#(221, 44, 0)
    colors_mask = ((1, 1, 1), (255, 255, 250), (255, 193, 0), (123, 31, 162), (2, 119, 189))
    colors_mask_mod = ((1, 1, 1), (255, 255, 250), (255, 193, 0), (123, 31, 162), (123, 31, 162))
    colors_mask = tuple(map(tuple, np.array(colors_mask)/255))
    colors_mask_mod = tuple(map(tuple, np.array(colors_mask_mod)/255))
    cmap_mask = LinearSegmentedColormap.from_list('Custom', colors_mask, len(colors_mask))
    cmap_mask_mod = LinearSegmentedColormap.from_list('Custom', colors_mask_mod, len(colors_mask_mod))
    axes[0, 0].imshow(img_main[slices[0]], label='IMG 10', cmap='gray')
    axes[1, 0].imshow(img_main[slices[1]], label='IMG 20', cmap='gray')
    axes[2, 0].imshow(img_main[slices[2]], label='IMG 35', cmap='gray')
    axes[3, 0].imshow(img_main[slices[3]], label='IMG 50', cmap='gray')
    axes[4, 0].imshow(img_main[slices[4]], label='IMG 70', cmap='gray')
    
    axes[0, 0].set_ylabel(f'Slice\n{slices[0]}', rotation=0, fontweight='bold', fontsize=12)
    axes[1, 0].set_ylabel(f'Slice\n{slices[1]}', rotation=0, fontweight='bold', fontsize=12)
    axes[2, 0].set_ylabel(f'Slice\n{slices[2]}', rotation=0, fontweight='bold', fontsize=12)
    axes[3, 0].set_ylabel(f'Slice\n{slices[3]}', rotation=0, fontweight='bold', fontsize=12)
    axes[4, 0].set_ylabel(f'Slice\n{slices[4]}', rotation=0, fontweight='bold', fontsize=12)
    
    axes[0, 0].set_title('Image\nSlices', fontsize=11, fontweight='bold')
    axes[0, 1].imshow(ground_truth_mask[slices[0]], label='GT 10', cmap=cmap_mask_mod)
    axes[1, 1].imshow(ground_truth_mask[slices[1]], label='GT 20', cmap=cmap_mask_mod)
    axes[2, 1].imshow(ground_truth_mask[slices[2]], label='GT 35', cmap=cmap_mask)
    axes[3, 1].imshow(ground_truth_mask[slices[3]], label='GT 50', cmap=cmap_mask)
    axes[4, 1].imshow(ground_truth_mask[slices[4]], label='GT 70', cmap=cmap_mask_mod)
    axes[0, 1].set_title('Ground\nTruth\n(Mask)', fontsize=11, rotation=0, ha='center', fontweight='bold')
    ######
    class_name = ['Back.', 'Shell', 'Yolk', 'Albumen', 'Air C.']
    j=2 
    for i in range(2,(2*n_classes)+2):
        if i%2==0:
            axes[0, i].imshow(predicted_mask_interv1[i-j][slices[0]], label='PM 10')
            axes[1, i].imshow(predicted_mask_interv1[i-j][slices[1]], label='PM 20')
            axes[2, i].imshow(predicted_mask_interv1[i-j][slices[2]], label='PM 35')
            axes[3, i].imshow(predicted_mask_interv1[i-j][slices[3]], label='PM 50')
            axes[4, i].imshow(predicted_mask_interv1[i-j][slices[4]], label='PM 70')
            # axes[0, i].set_title(f'Class {i-1}\nProbability\nOutput', fontsize=12, rotation=0, ha='center', fontweight='bold')
            axes[0, i].set_title(f'{class_name[i-j]}\nClass\nProb.\n(U-Net)\nOutput', fontsize=11, rotation=0, ha='center', fontweight='bold')
            
        else :
            axes[0, i].imshow(predicted_mask_interv2[i-j][slices[0]], label='PM 10')
            axes[1, i].imshow(predicted_mask_interv2[i-j][slices[1]], label='PM 20')
            axes[2, i].imshow(predicted_mask_interv2[i-j][slices[2]], label='PM 35')
            axes[3, i].imshow(predicted_mask_interv2[i-j][slices[3]], label='PM 50')
            axes[4, i].imshow(predicted_mask_interv2[i-j][slices[4]], label='PM 70')
            # axes[0, i].set_title(f'Class {i-1}\nProbability\nOutput', fontsize=12, rotation=0, ha='center', fontweight='bold')
            axes[0, i].set_title(f'{class_name[i-j]}\nClass\nProb.\n(FCN)\nOutput', fontsize=11, rotation=0, ha='center', fontweight='bold')
            j -= 1
        j += 1
    #######
    axes[0, i+1].imshow(outs_arg1[slices[0]], label='PM 10', cmap=cmap_mask_mod)
    axes[1, i+1].imshow(outs_arg1[slices[1]], label='PM 20', cmap=cmap_mask_mod)
    axes[2, i+1].imshow(outs_arg1[slices[2]], label='PM 35', cmap=cmap_mask)
    axes[3, i+1].imshow(outs_arg1[slices[3]], label='PM 50', cmap=cmap_mask)
    axes[4, i+1].imshow(outs_arg1[slices[4]], label='PM 70', cmap=cmap_mask_mod)
    axes[0, i+1].set_title('(U-Net)\nPred.\nMask\n(Argmax)', fontsize=11, rotation=0, ha='center', fontweight='bold')

    axes[0, i+2].imshow(outs_arg2[slices[0]], label='PM 10', cmap=cmap_mask_mod)
    axes[1, i+2].imshow(outs_arg2[slices[1]], label='PM 20', cmap=cmap_mask_mod)
    axes[2, i+2].imshow(outs_arg2[slices[2]], label='PM 35', cmap=cmap_mask)
    axes[3, i+2].imshow(outs_arg2[slices[3]], label='PM 50', cmap=cmap_mask)
    axes[4, i+2].imshow(outs_arg2[slices[4]], label='PM 70', cmap=cmap_mask_mod)
    axes[0, i+2].set_title('(FCN)\nPred.\nMask\n(Argmax)', fontsize=11, rotation=0, ha='center', fontweight='bold')

    axes[0, i+2].set_ylabel(f'Air\nChamber', rotation=270, fontweight='bold', fontsize=12)
    axes[1, i+2].set_ylabel(f'Albumen', rotation=270, fontweight='bold', fontsize=12)
    axes[2, i+2].set_ylabel(f'Yolk', rotation=270, fontweight='bold', fontsize=12)
    axes[3, i+2].set_ylabel(f'Shell', rotation=270, fontweight='bold', fontsize=12)
    axes[4, i+2].set_ylabel(f'Background', rotation=270, fontweight='bold', fontsize=12)

    axes[0, i+2].yaxis.set_label_coords(1.25,0.6)
    axes[1, i+2].yaxis.set_label_coords(1.25,0.6)
    axes[2, i+2].yaxis.set_label_coords(1.25,0.5)
    axes[3, i+2].yaxis.set_label_coords(1.25,0.5)
    axes[4, i+2].yaxis.set_label_coords(1.25,0.45)
    
    # colorbar = axes[3, i+2].collections[0].colorbar
    # colorbar.set_ticks([-0.667, 0, 0.667])
    # colorbar.set_ticklabels(['Backg.', 'Shell', 'Yolk', 'Alb.', 'Air.'])
    # plt.title('Predicted Mask')
    # fig.suptitle('Ground Truth vs Predicted Mask', fontsize=16, fontweight="bold")
    #.#-------------------------------------------------------------------------
    sm0 = ScalarMappable(cmap='viridis')
    sm0.set_array([])
    cbar0 = ColorbarBase(ax=fig.add_axes([0.910, 0.1, 0.006, 0.8]), cmap='viridis') 
    cbar0.ax.tick_params(size=0)
    cbar0.set_ticks([])
    cbar0.set_label(label='Probabilities', size='large', weight='bold', labelpad=10, rotation=270)#y=1.05


    sm = ScalarMappable(cmap=cmap_mask)
    sm.set_array([])
    # Create colorbar
    cbar = ColorbarBase(ax=fig.add_axes([0.875, 0.1, 0.006, 0.8]), cmap=cmap_mask)

    # Set colorbar labels and ticks
    # Modify these according to your needs
    # cbar.set_label(['Label', 'Label'])
    # cbar.ax.yaxis.set_ticks_position('right')
    # cbar.ax.yaxis.set_label_position('right')
    cbar.ax.tick_params(size=0)
    cbar.set_ticks([])

    # Add legends to colorbar
    # Modify these according to your needs
    # cbar.ax.set_yticklabels(['Background', 'Shell', 'Yolk', 'Albumen', 'Air Chamber'], rotation=90)
    # cbar.ax.set_anchor    set_ticklabels(['Background', 'Shell', 'Yolk', 'Albumen', 'Air Chamber'], rotation=90)
    #.#-------------------------------------------------------------------------

    for y in range(0,5):
        axes[y,0].yaxis.set_label_coords(-0.65,0.4)
        for i in range(1,14):
            if y == 4:
                axes[y, i].tick_params(axis='y', which='both', length=0)
            else:
                axes[y, i].tick_params(axis='both', which='both', length=0, labelsize=12)
    
    
    # a = plt.imshow(outs_arg[slices])
    # plt.colorbar(a)
    plt.show()
