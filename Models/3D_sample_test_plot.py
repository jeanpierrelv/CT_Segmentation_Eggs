import torch
import numpy as np
import matplotlib.pyplot as plt
from torch3D import read_image_CT, read_mask_3D, volume_mask_3D, measurements_3D
from model_3D import UNet, FCN
from data_prepare_3D import seg_data

splits_folder = 'data/data_backup/real_data_dicom/mix_filter_data'
img_dir = 'data/data_backup/real_data_dicom/mix_filter_data/images-crop'
mask_dir = 'data/data_backup/real_data_dicom/mix_filter_data/masks-crop'
model_path = 'data/data_backup/real_data_dicom/mix_filter_data/model_state_0.pth'
train_loss = splits_folder + '/' + 'train_loss_0.dat'
val_loss = splits_folder + '/' + 'val_loss_0.dat'
train_met = splits_folder + '/' + 'train_met_0.dat' 
val_met = splits_folder + '/' + 'val_met_0.dat' 
n_classes = 5
volume_pixel = 1#0.011038363
length_voxel = 1#0.22265625
n=0
train_split = np.loadtxt(splits_folder+'/train'+str(n)+'.txt',dtype=str)
train_data = seg_data(train_split,img_dir,mask_dir)
train_data.set_crop()

val_split = np.loadtxt(splits_folder+'/val'+str(n)+'.txt',dtype=str)
val_data = seg_data(val_split,img_dir,mask_dir)
val_data.crop = train_data.crop
val_data.crop_dims = train_data.crop_dims

test_split = np.loadtxt(splits_folder+'/test'+str(n)+'.txt',dtype=str)
test_data = seg_data(test_split,img_dir,mask_dir)
test_data.crop = train_data.crop
test_data.crop_dims = train_data.crop_dims
print(len(test_data))

test_loader = test_data.get_loader(batch_size=len(test_data),shuffle=False)
for batch in test_loader:
    imgs, masks = batch
    
# model = FCN(1,5)    
model = UNet(num_filters=8,n_classes=n_classes)
checkpoint = torch.load(model_path)
model.load_state_dict(checkpoint)

model.eval()
outs = model(imgs)
outs1 = outs[0]
arg_outs1 = torch.argmax(outs1, dim=0)

# Assume your mask is a 3D array of shape (slices, height, width)
# and it contains class labels 0, 1, 2, 3, 4

# Set the number of classes
num_classes = 5

# Create a colormap with num_classes colors
# cmap = plt.cm.get_cmap('tab10', num_classes)
cmap = plt.cm.get_cmap('viridis', num_classes)


fig1 = plt.figure(figsize=(8,8))
columns = 10
rows = 4
for i in range(1, columns*rows + 1):
    img = masks[0][i-1]
    fig1.add_subplot(rows, columns, i)
    plt.imshow(img, cmap=cmap, vmin=0, vmax=num_classes-1)

fig2 = plt.figure(figsize=(8,8))
columns = 10
rows = 4
for i in range(1, columns*rows + 1):
    img = arg_outs1[i-1]
    fig2.add_subplot(rows, columns, i)
    plt.imshow(img, cmap=cmap, vmin=0, vmax=num_classes-1)


fig3 = plt.figure(figsize=(8,8))
data = []
# with open(train_loss, 'r') as file:
#     for line in file:
#         data.append(line.strip())
data_train_loss = np.loadtxt(train_loss)
data_val_loss = np.loadtxt(val_loss)
plt.plot(data_train_loss, lw=3)
plt.plot(data_val_loss,lw=1)
plt.show()
