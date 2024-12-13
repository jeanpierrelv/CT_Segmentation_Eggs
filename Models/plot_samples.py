from os import listdir
import numpy as np
from torch3D import read_image_CT, read_mask_3D
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.cm import ScalarMappable
from matplotlib.colorbar import ColorbarBase
from matplotlib import ticker

n_classes = 5
path = 'data/egg_simulate_v3/slices-crop'
samples = listdir(path)
sample_choose = []
img_acc = []
mask_acc = []
for i in range(0,5):
    idx_choose = np.random.randint(0,len(samples))
    aux = samples.pop(idx_choose)
    sample_choose.append(aux)  
    img_dir = 'data/egg_simulate_v3/slices-crop'
    mask_dir = 'data/egg_simulate_v3/masks-crop'
    img_main, _ = read_image_CT(img_dir, aux)
    img_acc.append(img_main)
    mask = read_mask_3D(mask_dir, aux)
    mask_acc.append(mask)

slices = [15,35,65]
fig, axes = plt.subplots(2, (2*len(sample_choose)), figsize=(20, 6), sharex=True, sharey=True)
plt.subplots_adjust(hspace=0, wspace=-0.5)
plt.xticks([10,45,70])
plt.yticks([10,70,130])
class_name = ['Back.', 'Shell', 'Yolk', 'Albumen', 'Air C.']
colors_mask = ((1, 1, 1), (255, 255, 250), (255, 193, 0), (123, 31, 162), (2, 119, 189))
colors_mask_mod = ((1, 1, 1), (255, 255, 250), (255, 193, 0), (123, 31, 162), (123, 31, 162))
colors_mask = tuple(map(tuple, np.array(colors_mask)/255))
colors_mask_mod = tuple(map(tuple, np.array(colors_mask_mod)/255))
cmap_mask = LinearSegmentedColormap.from_list('Custom', colors_mask, len(colors_mask))
cmap_mask_mod = LinearSegmentedColormap.from_list('Custom', colors_mask_mod, len(colors_mask_mod))
j=0 
for i in range(0,(2*len(sample_choose))):
    if i%2==0:
        axes[0, i].imshow(img_acc[i-j][slices[0]], cmap='gray')
        axes[1, i].imshow(img_acc[i-j][slices[1]], cmap='gray')
        # axes[2, i].imshow(img_acc[i-j][slices[2]], cmap='gray')
        # axes[3, i].imshow(img_acc[i-j][slices[3]], cmap='gray')
        # axes[4, i].imshow(img_acc[i-j][slices[4]], cmap='gray')
        # axes[0, i].set_title(f'Class {i-1}\nProbability\nOutput', fontsize=12, rotation=0, ha='center', fontweight='bold')
        axes[0, i].set_title(f'Egg\nSample{j+1}', fontsize=12, rotation=0, ha='center', fontweight='bold')
        
    else :
        axes[0, i].imshow(mask_acc[i-j][slices[0]], cmap=cmap_mask_mod)
        axes[1, i].imshow(mask_acc[i-j][slices[1]], cmap=cmap_mask)
        # axes[2, i].imshow(mask_acc[i-j][slices[2]], cmap='gray')
        # axes[3, i].imshow(mask_acc[i-j][slices[3]], cmap='gray')
        # axes[4, i].imshow(mask_acc[i-j][slices[4]], cmap='gray')
        # axes[0, i].set_title(f'Class {i-1}\nProbability\nOutput', fontsize=12, rotation=0, ha='center', fontweight='bold')
        axes[0, i].set_title(f'Egg\nMask\nSample{j}', fontsize=12, rotation=0, ha='center', fontweight='bold')
        j -= 1
    j += 1
    axes[0, 0].set_ylabel(f'Slice\n{slices[0]}', rotation=0, fontweight='bold', fontsize=13)
    axes[1, 0].set_ylabel(f'Slice\n{slices[1]}', rotation=0, fontweight='bold', fontsize=13)
    # axes[2, 0].set_ylabel(f'Slice\n{slices[2]}', rotation=0, fontweight='bold', fontsize=12)

axes[1, i-4].set_ylabel(f'Air\nChamber', rotation=270, fontweight='bold', fontsize=12)
axes[1, i-3].set_ylabel(f'Albumen', rotation=270, fontweight='bold', fontsize=12)
axes[1, i-2].set_ylabel(f'Yolk', rotation=270, fontweight='bold', fontsize=12)
axes[1, i-1].set_ylabel(f'Shell', rotation=270, fontweight='bold', fontsize=12)
axes[1, i].set_ylabel(f'Background', rotation=270, fontweight='bold', fontsize=12)

axes[1, i-4].yaxis.set_label_coords(4.8,1.85)
axes[1, i-3].yaxis.set_label_coords(3.9,1.42)
axes[1, i-2].yaxis.set_label_coords(3,1.05)
axes[1, i-1].yaxis.set_label_coords(2.075,0.6)
axes[1, i].yaxis.set_label_coords(1.175,0.16)

sm = ScalarMappable(cmap=cmap_mask)
sm.set_array([])
# Create colorbar
cbar = ColorbarBase(ax=fig.add_axes([0.875, 0.1, 0.006, 0.8]), cmap=cmap_mask)
cbar.set_ticks([])
cbar.set_label(label='', size='large', weight='bold', labelpad=10, rotation=270)

plt.show()