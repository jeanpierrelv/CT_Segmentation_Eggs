import torch
import matplotlib.pyplot as plt
from model_3D import UNet, FCN, MaskRCNN, ConvNeXt
from torch3D import read_image_CT, read_mask_3D
import numpy as np

import matplotlib.cm as cm

img_dir = 'data/egg_simulate_v3/slices-crop'
mask_dir = 'data/egg_simulate_v3/masks-crop'
aux = 'a5no8uez3s'#'gd2k3gottt'#'wws8ll1s5s'#'ss512nmkjq'#testv3 gd2k3gottt #testv2 myj5ix0cvv
model_path = 'results/convnext/epochs_3000_lr_00001/model_state_0.pth'
#'results/convnext/epochs_200_lr_00001/model_state_0.pth'
#'results/fcn/split_80_10_10/epochs_1000_lr_00001/model_state_0.pth'
#results/results_simulate_231023_100epochs_adam_0001_out3_5class_nonorm_with_metrics/model_state_0.pth'
n_classes = 5
slices = [15,25,35,50,65]

img_main, _ = read_image_CT(img_dir, aux)
img = img_main.unsqueeze(0) # Adding a Channel dimension and Batch dimension, for have 5D as the input requirement in Conv3D
img = img.unsqueeze(0) # Adding , for have 5D as the input requirement in Conv3D
img = img.float()


mask = read_mask_3D(mask_dir, aux)
# model = UNet(num_filters=8,n_classes=n_classes)
# model = FCN(1, 5)
model = ConvNeXt()
# model = MaskRCNN(5)

checkpoint = torch.load(model_path)
model.load_state_dict(checkpoint)

ground_truth_mask = mask
model.eval()
predicted_mask = model(img)

with torch.no_grad():
    predicted_mask = predicted_mask.squeeze(0)
    
    outs_arg = torch.argmax(predicted_mask,dim=0)
    predicted_mask_interv = predicted_mask

    # # _min, _max = torch.amin(outs_arg).numpy(), torch.amax(outs_arg).numpy()
    # fig, axes = plt.subplots(n_classes+3, 5, figsize=(8, 10), sharex=True, sharey=True)
    # plt.subplots_adjust(hspace=0, wspace=-0.5)
    # plt.xticks([0,40,85])
    # axes[0, 0].imshow(img_main[slices[0]], label='IMG 10')
    # axes[0, 1].imshow(img_main[slices[1]], label='IMG 20')
    # axes[0, 2].imshow(img_main[slices[2]], label='IMG 35')
    # axes[0, 3].imshow(img_main[slices[3]], label='IMG 50')
    # axes[0, 4].imshow(img_main[slices[4]], label='IMG 70')
    # axes[0, 0].set_title(f'Slice\n{slices[0]}')
    # axes[0, 1].set_title(f'Slice\n{slices[1]}')
    # axes[0, 2].set_title(f'Slice\n{slices[2]}')
    # axes[0, 3].set_title(f'Slice\n{slices[3]}')
    # axes[0, 4].set_title(f'Slice\n{slices[4]}')
    # axes[0, 0].set_ylabel('Image\nSlices', fontsize=12, rotation=0, ha='right')
    # axes[1, 0].imshow(ground_truth_mask[slices[0]], label='GT 10')
    # axes[1, 1].imshow(ground_truth_mask[slices[1]], label='GT 20')
    # axes[1, 2].imshow(ground_truth_mask[slices[2]], label='GT 35')
    # axes[1, 3].imshow(ground_truth_mask[slices[3]], label='GT 50')
    # axes[1, 4].imshow(ground_truth_mask[slices[4]], label='GT 70')
    # axes[1, 0].set_ylabel('Ground\nTruth', fontsize=12, rotation=0, ha='right')
    # ######
    # for i in range(2,n_classes+2):
    #     axes[i, 0].imshow(predicted_mask_interv[0][slices[0]], label='PM 10')
    #     axes[i, 1].imshow(predicted_mask_interv[0][slices[1]], label='PM 20')
    #     axes[i, 2].imshow(predicted_mask_interv[0][slices[2]], label='PM 35')
    #     axes[i, 3].imshow(predicted_mask_interv[0][slices[3]], label='PM 50')
    #     axes[i, 4].imshow(predicted_mask_interv[0][slices[4]], label='PM 70')
    #     axes[i, 0].set_ylabel(f'Class {i-1}\nProbability', fontsize=12, rotation=0, ha='right')
    #     # axes[i, 0].autoscale(False)
    #     # axes[i, 1].autoscale(False)
    #     # axes[i, 2].autoscale(False)
    #     # axes[i, 3].autoscale(False)
    #     # axes[i, 4].autoscale(False)
    # #######

    # axes[i+1, 0].imshow(outs_arg[slices[0]], label='PM 10')
    # axes[i+1, 1].imshow(outs_arg[slices[1]], label='4PM 20')
    # axes[i+1, 2].imshow(outs_arg[slices[2]], label='PM 35')
    # axes[i+1, 3].imshow(outs_arg[slices[3]], label='PM 50')
    # axes[i+1, 4].imshow(outs_arg[slices[4]], label='PM 70')
    # axes[i+1, 0].set_ylabel('Predicted\nMask', fontsize=12, rotation=0, ha='right')
    # # plt.title('Predicted Mask')
    # fig.suptitle('Ground Truth vs Predicted Mask', fontsize=16, fontweight="bold")
    # plt.show()
####################################################################################
    fig, axes = plt.subplots(5, n_classes+3, figsize=(10, 8), sharex=True, sharey=True)
    plt.subplots_adjust(hspace=0, wspace=-0.5)
    plt.xticks([10,45,80])
    plt.yticks([10,70,130])
    axes[0, 0].imshow(img_main[slices[0]], label='IMG 10')
    axes[1, 0].imshow(img_main[slices[1]], label='IMG 20')
    axes[2, 0].imshow(img_main[slices[2]], label='IMG 35')
    axes[3, 0].imshow(img_main[slices[3]], label='IMG 50')
    axes[4, 0].imshow(img_main[slices[4]], label='IMG 70')
    
    axes[0, 0].set_ylabel(f'Slice\n{slices[0]}', rotation=0, fontweight='bold', fontsize=12)
    axes[1, 0].set_ylabel(f'Slice\n{slices[1]}', rotation=0, fontweight='bold', fontsize=12)
    axes[2, 0].set_ylabel(f'Slice\n{slices[2]}', rotation=0, fontweight='bold', fontsize=12)
    axes[3, 0].set_ylabel(f'Slice\n{slices[3]}', rotation=0, fontweight='bold', fontsize=12)
    axes[4, 0].set_ylabel(f'Slice\n{slices[4]}', rotation=0, fontweight='bold', fontsize=12)
    
    axes[0, 0].set_title('Image\nSlices', fontsize=12, fontweight='bold')
    axes[0, 1].imshow(ground_truth_mask[slices[0]], label='GT 10')
    axes[1, 1].imshow(ground_truth_mask[slices[1]], label='GT 20')
    axes[2, 1].imshow(ground_truth_mask[slices[2]], label='GT 35')
    axes[3, 1].imshow(ground_truth_mask[slices[3]], label='GT 50')
    axes[4, 1].imshow(ground_truth_mask[slices[4]], label='GT 70')
    axes[0, 1].set_title('Ground\nTruth\n(Mask)', fontsize=12, rotation=0, ha='center', fontweight='bold')
    ######
    class_name = ['Background', 'Shell', 'Yolk', 'Albumen', 'Ar Chamber'] 
    for i in range(2,n_classes+2):
        axes[0, i].imshow(predicted_mask_interv[i-2][slices[0]], label='PM 10')
        axes[1, i].imshow(predicted_mask_interv[i-2][slices[1]], label='PM 20')
        axes[2, i].imshow(predicted_mask_interv[i-2][slices[2]], label='PM 35')
        axes[3, i].imshow(predicted_mask_interv[i-2][slices[3]], label='PM 50')
        axes[4, i].imshow(predicted_mask_interv[i-2][slices[4]], label='PM 70')
        # axes[0, i].set_title(f'Class {i-1}\nProbability\nOutput', fontsize=12, rotation=0, ha='center', fontweight='bold')
        axes[0, i].set_title(f'{class_name[i-2]}\nClass\nProbability\nOutput', fontsize=12, rotation=0, ha='center', fontweight='bold')
    #######
    
    axes[0, i+1].imshow(outs_arg[slices[0]], label='PM 10')
    axes[1, i+1].imshow(outs_arg[slices[1]], label='PM 20')
    axes[2, i+1].imshow(outs_arg[slices[2]], label='PM 35')
    axes[3, i+1].imshow(outs_arg[slices[3]], label='PM 50')
    axes[4, i+1].imshow(outs_arg[slices[4]], label='PM 70')
    axes[0, i+1].set_title('Predicted\nMask\n(Argmax)', fontsize=12, rotation=0, ha='center', fontweight='bold')
    # plt.title('Predicted Mask')
    # fig.suptitle('Ground Truth vs Predicted Mask', fontsize=16, fontweight="bold")
    for y in range(0,5):
        axes[y,0].yaxis.set_label_coords(-0.65,0.4)
        for i in range(1,8):
            if y == 4:
                axes[y, i].tick_params(axis='y', which='both', length=0)
            else:
                axes[y, i].tick_params(axis='both', which='both', length=0, labelsize=12)
    # a = plt.imshow(outs_arg[slices])
    # plt.colorbar(a)
    plt.show()
####################################################################################
# # Plot of Ground Truth vs Prediction Mask
#     fig, axes = plt.subplots(3, 5, figsize=(12, 6), sharex=True, sharey=True)
#     plt.subplots_adjust(hspace=0, wspace=-0.5)
#     plt.xticks([0,40,85])
#     axes[0, 0].imshow(img_main[slices[0]], label='IMG 10')
#     axes[0, 1].imshow(img_main[slices[1]], label='IMG 20')
#     axes[0, 2].imshow(img_main[slices[2]], label='IMG 35')
#     axes[0, 3].imshow(img_main[slices[3]], label='IMG 50')
#     axes[0, 4].imshow(img_main[slices[4]], label='IMG 70')
#     axes[0, 0].set_title(f'Slice\n{slices[0]}')
#     axes[0, 1].set_title(f'Slice\n{slices[1]}')
#     axes[0, 2].set_title(f'Slice\n{slices[2]}')
#     axes[0, 3].set_title(f'Slice\n{slices[3]}')
#     axes[0, 4].set_title(f'Slice\n{slices[4]}')
#     axes[0, 0].set_ylabel('Image\nSlices', fontsize=12, rotation=0, ha='right')
#     axes[1, 0].imshow(ground_truth_mask[slices[0]], label='GT 10')
#     axes[1, 1].imshow(ground_truth_mask[slices[1]], label='GT 20')
#     axes[1, 2].imshow(ground_truth_mask[slices[2]], label='GT 35')
#     axes[1, 3].imshow(ground_truth_mask[slices[3]], label='GT 50')
#     axes[1, 4].imshow(ground_truth_mask[slices[4]], label='GT 70')
#     axes[1, 0].set_ylabel('Ground\nTruth', fontsize=12, rotation=0, ha='right')
#     axes[2, 0].imshow(outs_arg[slices[0]], label='PM 10')
#     axes[2, 1].imshow(outs_arg[slices[1]], label='PM 20')
#     axes[2, 2].imshow(outs_arg[slices[2]], label='PM 35')
#     axes[2, 3].imshow(outs_arg[slices[3]], label='PM 50')
#     axes[2, 4].imshow(outs_arg[slices[4]], label='PM 70')
#     axes[2, 0].set_ylabel('Predicted\nMask', fontsize=12, rotation=0, ha='right')
#     fig.suptitle('Ground Truth vs Predicted Mask', fontsize=16, fontweight="bold")
#     plt.show()

