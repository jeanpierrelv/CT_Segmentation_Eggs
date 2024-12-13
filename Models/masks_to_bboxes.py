#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 29 16:48:18 2024

@author: jean
"""

import torch
import numpy as np
# import torchvision.transforms.functional as F
# from torchvision.utils import draw_segmentation_masks
from torchvision.ops import masks_to_boxes


def mask_to_boxes(masks):
    """
    This is a converter from masks 3D into bounding boxes
    inputs: 
        masks : format(batch, depth, height, width)
    outputs:
        masks_boxes_batch : List of boxes in batches, for example if batch = 5
                            and depth = 80.
                            Is a list of 5 items, each item contain 80 items of
                            ith bounding boxes in form of (X1,Y1,X2,Y2).
    """
    batches = masks.shape[0]
    depth_value = masks.shape[1]
    masks_boxes_batch = []
    labels_boxes_batch = []
    masks_boxes3d_batch = []
    labels_boxes3d_batch =[]
    boxes_batches = {}
    boxes3d_batches = {}
    for batch_num in range(0,batches):
        masks_depth = []
        labels_depth = []
        for d in range(0,depth_value):
    
            # img = inputs[batch][0][d]
            # # uint8
            # img_byte = img.type('torch.ByteTensor')
            # rgb_img = img_byte.repeat(1, 3, 1, 1)
            # rgb_img = rgb_img.squeeze()
    
            mask_x = masks[batch_num][d]
            # mask_x = mask_x.unsqueeze(0)
            
            obj_ids = torch.unique(mask_x)
            # Without the background class 0
            obj_ids = obj_ids[1:]
            # obj_bool = torch.ne(obj_ids,0)
            # obj_ids = obj_ids[obj_bool]
            # print(np.array(obj_ids))
    
            masks_obj = mask_x == obj_ids[:, None, None]
            #-------------------------
            # num_objs = len(obj_ids)
            # boxes_xyz = []
            # for i in range(num_objs):
            #     pos = np.where(masks_obj[i])
            #     xmin = np.min(pos[1])
            #     xmax = np.max(pos[1])
            #     ymin = np.min(pos[0])
            #     ymax = np.max(pos[0])
            #     boxes_xyz.append([xmin, ymin, xmax, ymax])
            
            #-------------------------
            boxes = masks_to_boxes(masks_obj)
            
            labels = obj_ids#torch.ones((num_objs,), dtype=torch.int64)
            
            # print(boxes.size())
            # print(boxes)
            masks_depth.append(boxes)
            labels_depth.append(labels)
        #-------------------------    
        # Gets the initial and final depth (z) coordinate for each class
        # a1, a2, a3, a4 ---> Classes
        a1_z1, a2_z1, a3_z1, a4_z1, a1_z2, a2_z2, a3_z2, a4_z2 = [],[],[],[],[],[],[],[]
        for i in range(len(labels_depth)):
            if 1 in labels_depth[i]:
                if a1_z1 == []:
                    a1_z1 = i
                a1_z2 = i
            if 2 in labels_depth[i]:
                if a2_z1 == []:
                    a2_z1 = i
                a2_z2 = i  
            if 3 in labels_depth[i]:
                if a3_z1 == []:
                    a3_z1 = i
                a3_z2 = i
            if 4 in labels_depth[i]:
                if a4_z1 == []:
                    a4_z1 = i
                a4_z2 = i    
        # Gets the largest x1,y1 and x2,y2 coordinates of the bounding boxes for each class
        a1,a2,a3,a4=[],[],[],[]
        for i in range(len(torch.cat(labels_depth))):
            if torch.cat(labels_depth)[i] == 1:
                a1.append((torch.cat(masks_depth)[i]).unsqueeze(0))
            elif torch.cat(labels_depth)[i] == 2:
                a2.append((torch.cat(masks_depth)[i]).unsqueeze(0))
            elif torch.cat(labels_depth)[i] == 3:
                a3.append((torch.cat(masks_depth)[i]).unsqueeze(0))
            elif torch.cat(labels_depth)[i] == 4:
                a4.append((torch.cat(masks_depth)[i]).unsqueeze(0))
        # Rearrange the full coordinates x1,y1,z1,x2,y2,z2 of the bounding boxes for each class
        a1_3d = [torch.min(torch.cat(a1)[:,0]), torch.min(torch.cat(a1)[:,1]), torch.tensor(a1_z1).to(dtype=torch.float32), torch.max(torch.cat(a1)[:,2]), torch.max(torch.cat(a1)[:,3]), torch.tensor(a1_z2).to(dtype=torch.float32)]
        a2_3d = [torch.min(torch.cat(a2)[:,0]), torch.min(torch.cat(a2)[:,1]), torch.tensor(a2_z1).to(dtype=torch.float32), torch.max(torch.cat(a2)[:,2]), torch.max(torch.cat(a2)[:,3]), torch.tensor(a2_z2).to(dtype=torch.float32)]
        a3_3d = [torch.min(torch.cat(a3)[:,0]), torch.min(torch.cat(a3)[:,1]), torch.tensor(a3_z1).to(dtype=torch.float32), torch.max(torch.cat(a3)[:,2]), torch.max(torch.cat(a3)[:,3]), torch.tensor(a3_z2).to(dtype=torch.float32)]
        a4_3d = [torch.min(torch.cat(a4)[:,0]), torch.min(torch.cat(a4)[:,1]), torch.tensor(a4_z1).to(dtype=torch.float32), torch.max(torch.cat(a3)[:,2]), torch.max(torch.cat(a4)[:,3]), torch.tensor(a4_z2).to(dtype=torch.float32)]
        a1_3d = torch.tensor(np.array(a1_3d)).unsqueeze(0)
        a2_3d = torch.tensor(np.array(a2_3d)).unsqueeze(0)
        a3_3d = torch.tensor(np.array(a3_3d)).unsqueeze(0)
        a4_3d = torch.tensor(np.array(a4_3d)).unsqueeze(0)
        boxes_3d = torch.concat((a1_3d, a2_3d, a3_3d, a4_3d))
        labels_3d = torch.tensor((1.,2.,3.,4.)).unsqueeze(0)
        
        masks_boxes3d_batch.append(boxes_3d.unsqueeze(0))
        labels_boxes3d_batch.append(labels_3d.unsqueeze(0))
        #-------------------------
        masks_boxes_batch.append(torch.concat(masks_depth)) # review the dimensions, make into list to conserve the true dimensions
        labels_boxes_batch.append(torch.concat(labels_depth))
    
    boxes_batches["boxes"] = masks_boxes_batch
    boxes_batches["labels"] = labels_boxes_batch
    boxes_batches["masks"] = masks
    
    
    boxes3d_batches["boxes"] = torch.cat(masks_boxes3d_batch)
    boxes3d_batches["labels"] = torch.cat(labels_boxes3d_batch)
    boxes3d_batches["masks"] = masks
            
    return boxes_batches, boxes3d_batches


#------------------------------------------------------------------------------
# def show(imgs):
#     if not isinstance(imgs, list):
#         imgs = [imgs]
#     fix, axs = plt.subplots(ncols=len(imgs), squeeze=False)
#     for i, img in enumerate(imgs):
#         img = img.detach()
#         img = F.to_pil_image(img)
#         axs[0, i].imshow(np.asarray(img))
#         axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
#------------------------------------------------------------------------------
# Plot images with segmentation separately
# drawn_masks = []
# for mask in masks_obj:
#     drawn_masks.append(draw_segmentation_masks(rgb_img, mask, alpha=0.8, colors="blue"))  
# show(drawn_masks)
# #------------------------------------------------------------------------------
# from torchvision.utils import draw_bounding_boxes
# drawn_boxes = draw_bounding_boxes(rgb_img, boxes, colors="red")
# show(drawn_boxes)