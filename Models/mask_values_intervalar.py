import numpy as np
import torch


def mask_intervalar(predicted_mask, num_classes):
    """
    Transforma o valor de cada pixel float para uma classe: 
    Número de classes = 3, 
    output = Tensor de classes para cada pixel
    """
    # pred_arr = np.array(predicted_mask)
    pred_arr = predicted_mask
    gap = (pred_arr.max() - pred_arr.min())/num_classes
    pred_copy = pred_arr#.copy()
    # Create a mask to identify elements in the specified range
    mask_values = (pred_arr.min() + 2*gap < pred_arr)
    # Change values within the specified range to the new value
    pred_copy[mask_values] = 0

    mask_values = (pred_arr.min() + 0.6*gap < pred_arr) & (pred_arr < pred_arr.min() + 2 *gap)
    # Change values within the specified range to the new value
    pred_copy[mask_values] = 2

    mask_values = (pred_arr.min() + 0.6 *gap >= pred_arr)
    # Change values within the specified range to the new value
    pred_copy[mask_values] = 1

    # predicted_mask_interv = torch.Tensor(pred_copy)

    return pred_copy# predicted_mask_interv


def one_hot_3D(masks, num_classes):
    # Create an empty one-hot encoding tensor
    one_hot = torch.zeros((num_classes,) + masks.shape)

    # Convert the mask to int64 (if not already)
    mask_3d = masks.to(torch.int64)

    # Use scatter to create the one-hot encoding
    masks_hot = one_hot.scatter_(0, (mask_3d.unsqueeze(0)), 1)
    masks_hot = np.transpose(masks_hot,(1,0,2,3,4))
    return masks_hot

def zero_one_encode(predicted_mask):
    mask_values = (0.5 <= predicted_mask)
    predicted_mask[mask_values] = 1
    mask_values = (0.5 > predicted_mask)
    predicted_mask[mask_values] = 0
    return predicted_mask