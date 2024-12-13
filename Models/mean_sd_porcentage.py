thickness_voxel_unet = [70, 50, 100, 77.78, 80, 75, 87.50, 100, 114.29, 87.50, 66.67]
height_voxel_unet = [100, 100, 100, 98.89, 101.18, 98.95, 98.88, 96.59, 98.90, 96.55, 97.75]
width_voxel_unet = [104.54, 100, 100, 101.54, 98.54, 101.47, 98.53, 100, 101.41, 101.52, 98.51]

thickness_out_unet = [106.72, 111.97, 78.31, 78.33, 58.61, 121.35, 78.35, 107.17, 62.75, 87.33, 58.33]
height_out_unet = [100.80, 99.05, 100.60, 97.37, 101.04, 98.32, 101.77, 101.12, 99.98, 101.27, 100.99]
width_out_unet = [99.62, 101.64, 98.09, 100.71, 97.97, 101.15, 101.29, 99.82, 95.80, 98.30, 101.81]


thickness_voxel_fcn = [90.00, 91.67, 87.5, 111.11, 80, 100, 112.50, 133.33, 142.86, 142.86, 88.897]
height_voxel_fcn = [101.10, 101.14, 98.91, 100, 102.35, 100.00, 97.75, 101.14, 98.90, 96.55, 97.75]
width_voxel_fcn = [98.57, 100, 100, 103.08, 100, 98.48, 98.53, 101.47, 101.41, 100, 100]

thickness_out_fcn = [121.05, 118.75, 108.87, 109.45, 97.67, 98.57, 108.71, 158.55, 108.33, 82.67, 89.14]
height_out_fcn = [107.79, 103.67, 107.69, 100.37, 98.75, 108.09, 107.94, 108.44, 107.75, 107.70, 105.93]
width_out_fcn = [107.23, 107.58, 111.12, 104.53, 97.34, 108.25, 108.47, 106.62, 103.66, 103.74, 107.61]





def mean_sd(list_data):
    mean = sum(list_data) / len(list_data) 
    variance = sum([((x - mean) ** 2) for x in list_data]) / len(list_data) 
    res = variance ** 0.5
    print("Mean of sample is : " + str(mean)) 
    print("Standard deviation of sample is : " + str(res)) 


print(f'thickness_voxel_unet {mean_sd(thickness_voxel_unet)}')
print(f'height_voxel_unet {mean_sd(height_voxel_unet)}') 
print(f'width_voxel_unet {mean_sd(width_voxel_unet)}')  

print(f'thickness_out_unet {mean_sd(thickness_out_unet)}')
print(f'height_out_unet {mean_sd(height_out_unet)}') 
print(f'width_out_unet {mean_sd(width_out_unet)}') 


print(f'thickness_voxel_fcn {mean_sd(thickness_voxel_fcn)}')
print(f'height_voxel_fcn {mean_sd(height_voxel_fcn)}') 
print(f'width_voxel_fcn {mean_sd(width_voxel_fcn)}')  

print(f'thickness_out_fcn {mean_sd(thickness_out_fcn)}')
print(f'height_out_fcn {mean_sd(height_out_fcn)}') 
print(f'width_out_fcn {mean_sd(width_out_fcn)}') 