import torch
import numpy as np
import matplotlib.pyplot as plt
from torch3D import read_image_CT, read_mask_3D, volume_mask_3D, measurements_3D
from model_3D import UNet, FCN, UNet_Multitask, FCN_Multitask
from data_prepare_3D import seg_data

# splits_folder = 'data/egg_simulate_v3'
# img_dir = 'data/egg_simulate_v3/slices-crop'
# mask_dir = 'data/egg_simulate_v3/masks-crop'
# aux = 'gd2k3gottt'#'wws8ll1s5s'#'ss512nmkjq'#testv3 gd2k3gottt #testv2 myj5ix0cvv
# model_path = 'results/fcn/split_80_10_10/epochs_1000_lr_00001/model_state_0.pth'
# model_path = 'results/unet/split_80_10_10/epochs_150/model_state_0.pth'
#'results/fcn/split_80_10_10/epochs_1000_lr_00001/model_state_0.pth'
#'results/unet/split_80_10_10/epochs_150/model_state_0.pth'

splits_folder = 'data/data_backup/real_data_dicom/mix_filter_data'
img_dir = 'data/data_backup/real_data_dicom/mix_filter_data/images-crop'
mask_dir = 'data/data_backup/real_data_dicom/mix_filter_data/masks-crop'
aux = 'gd2k3gottt'#'wws8ll1s5s'#'ss512nmkjq'#testv3 gd2k3gottt #testv2 myj5ix0cvv
model_path = 'data/data_backup/real_data_dicom/mix_filter_data/model_state_0.pth'

#'results/fcn/split_80_10_10/epochs_1000_lr_00001/model_state_0.pth'
#'results/fcn/split_80_10_10/epochs_200_lr_00001//model_state_0.pth'
#'results/unet/split_70_16_4/results_simulate_231023_100epochs_adam_0001_out3_5class_nonorm_with_metrics/model_state_0.pth'
n_classes = 5
pixel_height = 0.6503906
pixel_widh = 0.6503906
pixel_depth = 3
volume_pixel = pixel_height * pixel_widh * pixel_depth#0.32539073#1#0.011038363
length_voxel = 0.6503906#1#0.22265625
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
# exit()
samples_testset = len(test_split)
test_loader = test_data.get_loader(batch_size=len(test_data),shuffle=False)

for batch in test_loader:
    if len(batch) == 2:
      imgs, masks = batch
    elif len(batch) == 3:
      imgs, masks, measures = batch
    
# model = FCN(1,5)    
# model = FCN_Multitask(1,5)    
model = UNet(num_filters=8,n_classes=n_classes)
# model = UNet_Multitask(num_filters=8,n_classes=n_classes)
checkpoint = torch.load(model_path)
model.load_state_dict(checkpoint)


outs_volumes_acc =[]
masks_volumes_acc = []
accuracy_one_acc = []

model.eval()
outs = model(imgs)
multitask = False
if type(outs) is tuple:
  outs1 = outs[1]
  outs = outs[0]
  multitask = True
  
outs = torch.argmax(outs, dim=1)
for i in range(len(test_data)):
    outs_volumes = volume_mask_3D(outs[i])
    masks_volumes = volume_mask_3D(masks[i])
    outs_volumes_acc.append(outs_volumes)
    masks_volumes_acc.append(masks_volumes)
    accuracy_one = np.array(outs_volumes)/np.array(masks_volumes)
    accuracy_one_acc.append(accuracy_one)

mean_accuracy = (np.sum(np.array(accuracy_one_acc), 0)/samples_testset)*100
    
outs_acc = torch.tensor(np.array(outs_volumes_acc))
masks_acc = torch.tensor(np.array(masks_volumes_acc))

outs_shell = (torch.sum(outs_acc[:,:1]))
masks_shell = (torch.sum(masks_acc[:,:1]))
accuracy_shell = (outs_shell/masks_shell)*100

outs_yolk = (torch.sum(outs_acc[:,1:2]))
masks_yolk = (torch.sum(masks_acc[:,1:2]))
accuracy_yolk = (outs_yolk/masks_yolk)*100

outs_albumen = (torch.sum(outs_acc[:,2:3]))
masks_albumen = (torch.sum(masks_acc[:,2:3]))
accuracy_albumen = (outs_albumen/masks_albumen)*100

outs_air = (torch.sum(outs_acc[:,3:4]))
masks_air = (torch.sum(masks_acc[:,3:4]))
accuracy_air = (outs_air/masks_air)*100

outs_background = (torch.sum(outs_acc[:,-1:]))
masks_background = (torch.sum(masks_acc[:,-1:]))
accuracy_background = (outs_background/masks_background)*100

print(f"The total SHELL voxels in the test set output are: {outs_shell}")
print(f"The total SHELL voxels in the test set mask are: {masks_shell}")
print(f"The total SHELL VOLUME in the test set output are: {outs_shell*volume_pixel}")
print(f"The total AVERAGE SHELL VOLUME in the test set output are: {outs_shell*volume_pixel/samples_testset}")
print(f"The success percentage of the SHELL class in the test set: {accuracy_shell}")
print(f"The success MEAN percentage of the SHELL class in the test set: {mean_accuracy[0]}")
# print(f"The success REAL MEAN percentage of the SHELL class in the test set: {real_mean_accuracy_shell}")
print(f"The total YOLK voxels in the test set output are: {outs_yolk}")
print(f"The total YOLK voxels in the test set mask are: {masks_yolk}")
print(f"The total YOLK VOLUME in the test set output are: {outs_yolk*volume_pixel}")
print(f"The total AVERAGE YOLK VOLUME in the test set output are: {outs_yolk*volume_pixel/samples_testset}")
print(f"The success percentage of the YOLK class in the test set: {accuracy_yolk}")
print(f"The success MEAN percentage of the YOLK class in the test set: {mean_accuracy[1]}")
# print(f"The success REAL MEAN percentage of the YOLK class in the test set: {real_mean_accuracy_yolk}")
print(f"The total ALBUMEN voxels in the test set output are: {outs_albumen}")
print(f"The total ALBUMEN voxels in the test set mask are: {masks_albumen}")
print(f"The total ALBUMEN VOLUME in the test set output are: {outs_albumen*volume_pixel}")
print(f"The total AVERAGE ALBUMEN VOLUME in the test set output are: {outs_albumen*volume_pixel/samples_testset}")
print(f"The success percentage of the ALBUMEN class in the test set: {accuracy_albumen}")
print(f"The success MEAN percentage of the ALBUMEN class in the test set: {mean_accuracy[2]}")
# print(f"The success REAL MEAN percentage of the ALBUMEN class in the test set: {real_mean_accuracy_albumen}")
print(f"The total AIR CHAMBER voxels in the test set output are: {outs_air}")
print(f"The total AIR CHAMBER voxels in the test set mask are: {masks_air}")
print(f"The total AIR CHAMBER VOLUME in the test set output are: {outs_air*volume_pixel}")
print(f"The total AVERAGE AIR CHAMBER VOLUME in the test set output are: {outs_air*volume_pixel/samples_testset}")
print(f"The success percentage of the AIR CHAMBER class in the test set: {accuracy_air}")
print(f"The success MEAN percentage of the AIR CHAMBER class in the test set: {mean_accuracy[3]}")
# print(f"The success REAL MEAN percentage of the AIR CHAMBER class in the test set: {real_mean_accuracy_air}")
print(f"The total BACKGROUND voxels in the test set output are: {outs_background}")
print(f"The total BACKGROUND voxels in the test set mask are: {masks_background}")
print(f"The total BACKGROUND VOLUME in the test set output are: {outs_background*volume_pixel}")
print(f"The total AVERAGE BACKGROUND VOLUME in the test set output are: {outs_background*volume_pixel/samples_testset}")
print(f"The success percentage of the BACKGROUND class in the test set: {accuracy_background}")
print(f"The success MEAN percentage of the BACKGROUND class in the test set: {mean_accuracy[4]}")
# print(f"The success REAL MEAN percentage of the BACKGROUND class in the test set: {real_mean_accuracy_background}")




import numpy as np

def calculate_metrics(masks, predictions):
  """
  This function compares ground truth masks with predicted masks and calculates
  accuracy, precision, recall for each class per image ID.

  Args:
      masks: A numpy array of shape (num_images, depth, height, width) containing ground truth masks.
      predictions: A numpy array of shape (num_images, depth, height, width) containing predicted masks.

  Returns:
      A dictionary containing average accuracy, precision, and recall for each class.
  """
  class_labels = {0: "Background", 1: "Shell", 2: "Yolk", 3: "Albumen", 4: "Air Chamber"}
  num_images, depth, height, width = masks.shape
  num_classes = len(class_labels)

  # Initialize dictionaries to store results
  accuracy_per_image = {}
  precision_per_class = {cls: [] for cls in class_labels.keys()}
  recall_per_class = {cls: [] for cls in class_labels.keys()}
  f1_per_class = {cls: [] for cls in class_labels.keys()}
  matthews_per_class = {cls: [] for cls in class_labels.keys()}
  kappa_per_class = {cls: [] for cls in class_labels.keys()}

  for image_id in range(num_images):
    mask = masks[image_id]
    prediction = predictions[image_id]

    # Calculate intersection over union (IOU) for each class
    precision_per_class_r = np.zeros(num_classes)
    recall_per_class_r = np.zeros(num_classes)
    f1_per_class_r = np.zeros(num_classes)
    matthews_per_class_r = np.zeros(num_classes)
    kappa_per_class_r = np.zeros(num_classes)
    iou_per_class = np.zeros(num_classes)
    for cls in class_labels.keys():
      positives = np.sum((mask == cls))
      negatives = (depth * height * width) - np.sum((mask == cls))
      true_positives = np.sum((mask == cls) & (prediction == cls))
      false_positives = np.sum((prediction == cls) & (mask != cls))
      true_negatives = np.sum((mask != cls) & (prediction != cls))
      false_negatives = np.sum((mask == cls) & (prediction != cls))
      predicted_positive = np.sum((prediction == cls))
      predicted_negatives = np.sum((prediction != cls))
      iou = true_positives / (true_positives + false_positives + false_negatives + 1e-7)
      precision_r = true_positives / (true_positives + false_positives + 1e-7)
      recall_r = true_positives / (true_positives + false_negatives + 1e-7)
      f1_r = (2*true_positives) / ((2*true_positives) + false_positives + false_negatives + 1e-7)
      tpr = true_positives/positives
      tnr = true_negatives/negatives
      ppv = true_positives/predicted_positive
      npv = true_negatives/predicted_negatives
      fnr = false_negatives/positives
      fpr = false_positives/negatives
      for1 = false_negatives/predicted_negatives
      fdr = false_positives/predicted_positive
      matthews_r = np.sqrt(tpr * tnr * ppv * npv) - np.sqrt(fnr * fpr * for1 * fdr) + 1e-7
      kappa_r = (2 * (true_positives * true_negatives - false_negatives * false_positives))/((true_positives + false_positives) * (false_positives + true_negatives) + (true_positives + false_negatives) * (false_negatives + true_negatives))
      precision_per_class_r[cls] = precision_r
      recall_per_class_r[cls] = recall_r
      f1_per_class_r[cls] = f1_r
      matthews_per_class_r[cls] = matthews_r
      kappa_per_class_r[cls] = kappa_r
      iou_per_class[cls] = iou

    # Calculate accuracy per image
    total_pixels = depth * height * width
    correct_pixels = np.sum(mask == prediction)
    accuracy = correct_pixels / total_pixels
    accuracy_per_image[image_id] = accuracy

    # Calculate precision and recall for each class
    for cls in class_labels.keys():
      # precision = iou_per_class[cls] if np.sum(prediction == cls) > 0 else 0
      # recall = iou_per_class[cls] if np.sum(mask == cls) > 0 else 0
      precision = precision_per_class_r[cls] if np.sum(prediction == cls) > 0 else 0
      recall = recall_per_class_r[cls] if np.sum(mask == cls) > 0 else 0
      f1 = f1_per_class_r[cls] if np.sum(mask == cls) > 0 else 0
      matthews = matthews_per_class_r[cls] if np.sum(mask == cls) > 0 else 0
      kappa = kappa_per_class_r[cls] if np.sum(mask == cls) > 0 else 0
      precision_per_class[cls].append(precision)
      recall_per_class[cls].append(recall)
      f1_per_class[cls].append(f1)
      matthews_per_class[cls].append(matthews)
      kappa_per_class[cls].append(kappa)

  # Calculate average metrics
  average_accuracy = np.mean(list(accuracy_per_image.values()))
  average_precision = {cls: np.mean(values) for cls, values in precision_per_class.items()}
  average_recall = {cls: np.mean(values) for cls, values in recall_per_class.items()}
  average_f1 = {cls: np.mean(values) for cls, values in f1_per_class.items()}
  average_matthews = {cls: np.mean(values) for cls, values in matthews_per_class.items()}
  average_kappa = {cls: np.mean(values) for cls, values in kappa_per_class.items()}

  # Return dictionary with all metrics
  return {
      "average_accuracy": average_accuracy,
      "average_precision": average_precision,
      "average_recall": average_recall,
      "average_f1": average_f1,
      "average_matthews": average_matthews,
      "average_kappa": average_kappa,
  }

# Example usage
# masks = np.random.randint(0, 5, size=(3, 10, 20, 25))  # Example mask data
# predictions = np.random.randint(0, 5, size=(3, 10, 20, 25))  # Example prediction data
metrics = calculate_metrics(np.array(masks), np.array(outs))
class_labels = {0: "Background", 1: "Shell", 2: "Yolk", 3: "Albumen", 4: "Air Chamber"}
print(f"Average Accuracy: {metrics['average_accuracy']:.4f}")
g=[]
for cls, value in metrics["average_precision"].items():
  print(f"Average Precision ({class_labels[cls]}): {value:.4f}")
  g.append(value)
print(f"Total Average Precision: {value:.4f}")
g=[]
for cls, value in metrics["average_recall"].items():
  print(f"Average Recall ({class_labels[cls]}): {value:.4f}")
  g.append(value)
print(f"Total Average Recall: {value:.4f}")
g=[]
for cls, value in metrics["average_f1"].items():
  print(f"Average F1-Score ({class_labels[cls]}): {value:.4f}")
  g.append(value)
print(f"Total Average F1-Score: {value:.4f}")
g=[]
for cls, value in metrics["average_matthews"].items():
  print(f"Average Matthews Correlation ({class_labels[cls]}): {value:.4f}")
  g.append(value)
print(f"Total Average Matthews Correlation: {value:.4f}")
g=[]
for cls, value in metrics["average_kappa"].items():
  print(f"Average Kappa Score ({class_labels[cls]}): {value:.4f}")
  g.append(value)
print(f"Total Average Kappa Score: {value:.4f}")



thickness_shells_outs, heights_outs, widths_outs = np.array(measurements_3D(outs))*length_voxel
thickness_shells_masks, heights_masks, widths_masks = np.array(measurements_3D(masks))*length_voxel




print(f"Egg heights from the test dataset - Masks: {heights_masks}")
print(f"Egg heights from the test dataset - Outs: {heights_outs}")
print(f"Egg Widths from the test dataset - Masks: {widths_masks}")
print(f"Egg Widths from the test dataset - Outs: {widths_outs}")
print(f"Egg shell thickness from the test dataset - Masks: {thickness_shells_masks}")
print(f"Egg shell thickness from the test dataset - Outs: {thickness_shells_outs}")
if multitask is True:
  mse = torch.nn.MSELoss()
  mse_result = mse(outs1, measures)
  print(f"Real Measures - Masks (Thickness - Height - Width): \n {measures} \n")
  print(f"Real Measures estimated- output network (Thickness - Height - Width): \n {outs1} \n")
  print(f"MSE - Test Set: \n {mse_result} \n")
