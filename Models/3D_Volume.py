import torch
import numpy as np
import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go
from torch3D import read_image_CT, read_mask_3D
from model_3D import UNet
from scipy.ndimage import measurements, center_of_mass, binary_dilation, zoom

img_dir = 'data/egg_simulate_v3/slices-crop'
mask_dir = 'data/egg_simulate_v3/masks-crop'
aux = 'gd2k3gottt'#'7d22ho27tn'#testv3 gd2k3gottt #testv2 myj5ix0cvv
model_path = 'results/results_simulate_231023_100epochs_adam_0001_out3_5class_nonorm_with_metrics/model_state_0.pth'

img = read_image_CT(img_dir, aux)
img = img.unsqueeze(0) # Adding a Channel dimension and Batch dimension, for have 5D as the input requirement in Conv3D
img = img.unsqueeze(0) # Adding , for have 5D as the input requirement in Conv3D
img = img.float()

mask = read_mask_3D(mask_dir, aux)
model = UNet(8,n_classes=5)

checkpoint = torch.load(model_path)
model.load_state_dict(checkpoint)

ground_truth_mask = mask
model.eval()
predicted_mask = model(img)

# predicted_mask = (predicted_mask - predicted_mask.min())/(predicted_mask.max() - predicted_mask.min())
# predicted_mask = predicted_mask * 255
img = img.squeeze(0,1)


mask_data_1 = ground_truth_mask
mask_data_1 = zoom(1*(mask_data_1), (0.4,0.4,0.4))
z, y, x = [np.arange(i) for i in mask_data_1.shape]
X,Y,Z = np.meshgrid(x,y,z, indexing='ij')
# Z*=4

color_mask_1 = 'viridis'#[[0, 'green'], [0.5, 'red'], [1.0, 'rgb(255, 0, 0)']]  # Red color for mask 1'blues' #[[0, 'green'], [0.5, 'red'], [1.0, 'rgb(255, 0, 0)']]  # Red color for mask 1
color_mask_2 = 'oranges' #[[0, 'white'], [0.5, 'yellow'], [1.0, 'rgb(255, 255, 204)']]  # Yellow color for mask 2

trace_mask_1 = go.Volume(
    x=X.flatten(),
    y=Y.flatten(),
    z=Z.flatten(),
    value=np.transpose(mask_data_1,(2,1,0)).flatten(),
    isomin=0.1,  # Minimum isosurface value
    # isomax=1,  # Maximum isosurface value
    opacity=0.1,  # Opacity of the mask
    surface_count=17,  # Number of isosurfaces
    colorscale=color_mask_1,
    # showscale=False  # Hide the color scale legend
)

with torch.no_grad():
    predicted_mask = predicted_mask.squeeze(0)
    # predicted_mask = (predicted_mask - predicted_mask.min())/(predicted_mask.max() - predicted_mask.min())
    outs_arg = torch.argmax(predicted_mask,dim=0)
    mask_data_2 = outs_arg
    mask_data_2 = zoom(1*(mask_data_2), (0.4,0.4,0.4))
    value_aux = np.transpose(mask_data_2,(2,1,0)).flatten() 
    z2, y2, x2 = [np.arange(i) for i in mask_data_2.shape]
    X2,Y2,Z2 = np.meshgrid(x2,y2,z2, indexing='ij')
    trace_mask_2 = go.Volume(
        x=X2.flatten(),
        y=Y2.flatten(),
        z=Z2.flatten(),
        value=value_aux,
        isomin=0.1,
        # isomax=1,
        opacity=0.1,
        surface_count=17,
        colorscale=color_mask_2,
        # showscale=False
    )
    
fig = go.Figure(data=[trace_mask_2, trace_mask_1])
fig.write_html("GT_and_PM_10batch_100epochs_0001_5class_simulatev3.html")

# fig.update_layout(
#     scene=dict(aspectmode="cube"),  # Set aspect mode for a cube-shaped plot
#     title="3D Volume Masks",
    # margin=dict(l=0, r=0, b=0, t=0)  # Adjust margins as needed
# )

# fig.show()



# fig = plt.figure()
# fig = go.Figure(data=go.Volume(
#     x=X.flatten(),
#     y=Y.flatten(),
#     z=Z.flatten(),
#     value=np.transpose(ground_truth_mask,(1,2,0)).flatten(),
#     isomin=0.1,
#     opacity=0.1, # needs to be small to see through all surfaces
#     surface_count=17, # needs to be a large number for good volume rendering
#     ))
# fig.show()

