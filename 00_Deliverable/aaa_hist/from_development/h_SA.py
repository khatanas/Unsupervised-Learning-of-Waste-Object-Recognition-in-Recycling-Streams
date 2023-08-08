import os
from os.path import join
os.chdir('/home/dkhatanassia/segment-anything')
from segment_anything import sam_model_registry, SamPredictor, SamAutomaticMaskGenerator

import numpy as np
from copy import deepcopy
from pycocotools import mask as maskUtils
from skimage.measure import label
import matplotlib.pyplot as plt
import torch
import json
import cv2
import random

from h_json import readJson

############### Directly from {SA,detectron2} github or slighty modified ########################

def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:,:,3] = 0
    random.seed(10)
    for ann in sorted_anns:
        m = ann['segmentation']
        random.seed(10)
        color_mask = np.concatenate([np.random.random(3), [0.5]])
        img[m] = color_mask
    ax.imshow(img)

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    
def show_points(coords, labels, ax, marker_size=20):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='o', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='o', s=marker_size, edgecolor='white', linewidth=1.25)   

def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))    

def batched_mask_to_box(masks: torch.Tensor) -> torch.Tensor:
    """
    Calculates boxes in XYXY format around masks. Return [0,0,0,0] for
    an empty mask. For input shape C1xC2x...xHxW, the output shape is C1xC2x...x4.
    """
    # torch.max below raises an error on empty inputs, just skip in this case
    if torch.numel(masks) == 0:
        return torch.zeros(*masks.shape[:-2], 4, device=masks.device)

    # Normalize shape to CxHxW
    shape = masks.shape
    h, w = shape[-2:]
    if len(shape) > 2:
        masks = masks.flatten(0, -3)
    else:
        masks = masks.unsqueeze(0)

    # Get top and bottom edges
    in_height, _ = torch.max(masks, dim=-1)
    in_height_coords = in_height * torch.arange(h, device=in_height.device)[None, :]
    bottom_edges, _ = torch.max(in_height_coords, dim=-1)
    in_height_coords = in_height_coords + h * (~in_height)
    top_edges, _ = torch.min(in_height_coords, dim=-1)

    # Get left and right edges
    in_width, _ = torch.max(masks, dim=-2)
    in_width_coords = in_width * torch.arange(w, device=in_width.device)[None, :]
    right_edges, _ = torch.max(in_width_coords, dim=-1)
    in_width_coords = in_width_coords + w * (~in_width)
    left_edges, _ = torch.min(in_width_coords, dim=-1)

    # If the mask is empty the right edge will be to the left of the left edge.
    # Replace these boxes with [0, 0, 0, 0]
    empty_filter = (right_edges < left_edges) | (bottom_edges < top_edges)
    out = torch.stack([left_edges, top_edges, right_edges, bottom_edges], dim=-1)
    out = out * (~empty_filter).unsqueeze(-1)

    # Return to original shape
    if len(shape) > 2:
        out = out.reshape(*shape[:-2], 4)
    else:
        out = out[0]

    return out

def box_xyxy_to_xywh(box_xyxy: torch.Tensor) -> torch.Tensor:
    box_xywh = deepcopy(box_xyxy)
    box_xywh[2] = box_xywh[2] - box_xywh[0]
    box_xywh[3] = box_xywh[3] - box_xywh[1]
    return box_xywh

############### perso ########################
def initSam():
    path2cp = '/home/dkhatanassia/segment-anything'
    sam_checkpoint = "sam_vit_h_4b8939.pth"
    model_type = "vit_h"
    device = "cuda"
    sam = sam_model_registry[model_type](checkpoint=join(path2cp,sam_checkpoint))
    sam.to(device=device)
    return sam


def encodeMasks(masks):
    tojson = deepcopy(masks)
    for i in range(len(masks)):
        binary_mask = masks[i]['segmentation']
        rle = maskUtils.encode(np.asfortranarray(binary_mask))
        rle['counts'] = rle['counts'].decode('utf-8')
        tojson[i]['segmentation'] = rle
    return tojson


def decodeMasks(masks):
    for i in range(len(masks)):
        rle = masks[i]['segmentation']
        #rle['counts']= rle['counts'].encode('utf-8')
        masks[i]['segmentation'] = maskUtils.decode(rle).astype(bool)
    return masks
    
    
def getMask(image, predictor, xy_list,label_list,show=False,p=5):
    '''
    Takes as input the image, the predictor, the prompts
    Returns the resulting mask together with info
    '''
    input_point = np.array(xy_list)
    input_label = np.array(label_list)

    mask, score, logit = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        multimask_output=False
        )
    
    if show:
        plt.figure(figsize=(p,p))
        plt.imshow(image)
        show_mask(mask, plt.gca())
        show_points(input_point, input_label, plt.gca())
        plt.title(f"best mask, score: {float(score):.2f}", fontsize=14)
        plt.axis('on')
        plt.show()  

    return mask.squeeze(), score, logit


def IoU(mask1,mask2):
    segm1 = mask1['segmentation']
    segm2 = mask2['segmentation']
    
    inter = np.logical_and(segm1,segm2).sum()
    union = np.logical_or(segm1,segm2).sum()
    return inter/union


def maskOverlay2(masks, th_IoU=0.015):
    '''
    returns a list of list, sorted by area. contains the idx of the masks overlapping with each other
    '''
        
    sorted_annotations = sorted(masks, key=lambda x: x['area'],reverse=True)
    
    are_overlapping = []
    to_skip = []
    for ann_i in sorted_annotations:
        if ann_i['id'] in to_skip: continue
        segm_i=ann_i['segmentation']
        tmp=[ann_i['id']]
        to_skip.append(ann_i['id'])
        
        for ann_j in sorted_annotations:
            if ann_j['id'] in to_skip: continue
            segm_j = ann_j['segmentation']
            if np.logical_and(segm_i,segm_j).any():
                if IoU(ann_i,ann_j)>th_IoU: 
                    tmp.append(ann_j['id'])
                    to_skip.append(ann_j['id'])
                    
        are_overlapping.append(tmp)
    
    return sorted(are_overlapping, key = lambda x: x[0])


def updateBbox(mask):
    rle = maskUtils.encode(np.asfortranarray(mask.astype(np.uint8)))
    bbox = maskUtils.toBbox(rle)
    bbox = [int(b) for b in bbox]
    return bbox


def findDuplicates(masks,th_duplicate=.95):
    best_c_id = [[] for _ in range(len(masks))]
    pass_list = []    
    
    are_overlapping = maskOverlay2(masks)
    
    # init container
    for nb, mask in enumerate(masks):
        assert mask['id']==nb
        if nb in pass_list:continue
        matching_sublists = [sublist for sublist in are_overlapping if nb in sublist]
        matching_anns = [item for sublist in matching_sublists for item in sublist]

        for maskbar_id in matching_anns:
            if (maskbar_id==nb) or (maskbar_id in pass_list): continue
            maskbar = masks[maskbar_id]
            if np.logical_and(mask['segmentation'],maskbar['segmentation']).any():
                if IoU(mask,maskbar) > th_duplicate:
                    best_c_id[mask['id']].append(maskbar['id'])
                    pass_list.append(maskbar['id'])

    is_a_match = []
    for nb, mask in enumerate(masks):
        assert nb == mask['id']
        if len(best_c_id[nb])>0: 
            is_a_match.append([nb,[id for id in best_c_id[nb]]])
            
    return is_a_match


def cleanDuplicates(masks, th_duplicate=0.95):
    duplicates = findDuplicates(masks,th_duplicate=th_duplicate)
    to_be_removed_id = [item for pair in duplicates for item in pair[1]]
    for pair in duplicates:
        for duplicate_id in pair[1]:
            id0 = pair[0]
            id1 = duplicate_id
            ann = masks[id0]
            ann['segmentation']=np.logical_or(ann['segmentation'],masks[id1]['segmentation'])
        ann['area'] = int(ann['segmentation'].sum())
        ann['bbox'] = updateBbox(ann['segmentation'])
        
    masks = [mask for mask in masks if mask['id'] not in to_be_removed_id]
    
    for idx,mask in enumerate(masks):
        mask['id'] = idx
        
    return masks


def cleanMasks(masks,th_area=1000):
    for idx,ann in enumerate(masks):
        ann['id'] = idx
        token=0
        
        # count separate items
        tmp = label(ann['segmentation'])
        # if more than one
        if tmp.max()>1:
            # check area and remove if #pixels < threshold
            for i in range(1, len(np.unique(tmp))):
                if (tmp==i).sum()<th_area: 
                    ann['segmentation'][tmp==i]=0
                    token = 1
            if token:
                ann['area'] = int(ann['segmentation'].sum())
                ann['bbox'] = updateBbox(ann['segmentation'])
                    
    masks = cleanDuplicates(masks)
    
    return masks




'''def cleanMasks(masks,th_area=1000, th_overlay=0.95):

    for idx,mask in enumerate(masks):
        mask['id'] = idx
        
    # find overlaying masks
    are_overlapping = maskOverlay2(masks)
    
    # clean
    bad_duplicates = []
    for list_ in are_overlapping:
        modified_id=[]
        for m_id in list_:
            token=0
            ann = masks[m_id]
            # count separate items
            tmp = label(ann['segmentation'])
            # if more than one
            if tmp.max()>1:
                # check area and remove if #pixels < threshold
                for i in range(1, len(np.unique(tmp))):
                    if (tmp==i).sum()<th_area: 
                        ann['segmentation'][tmp==i]=0
                        token = 1
                if token:
                    ann['area'] = int(ann['segmentation'].sum())
                    ann['bbox'] = updateBbox(ann['segmentation'])
                    modified_id.append(m_id)
                        
        for mod_id in modified_id:
            for m_id in list_:
                ann = masks[m_id]
                if mod_id==m_id:continue
                elif IoU(masks[mod_id],ann)>th_overlay:
                    tmp_remove = mod_id if masks[mod_id]['stability_score']<ann['stability_score'] else m_id
                    bad_duplicates.append(tmp_remove)
                    break
                
    filtered = [masks[i] for i in range(len(masks)) if not i in bad_duplicates]
    for idx,mask in enumerate(filtered):
        mask['id'] = idx

    return filtered'''


def addBorder(image, border_width):
    '''
    Add a border of {border_width} around the image
    '''
    height, width, channels = image.shape
    new_height = height + 2 * border_width
    new_width = width + 2 * border_width
    bordered_image = np.zeros((new_height, new_width, channels), dtype=image.dtype)
    bordered_image[border_width : border_width + height, border_width : border_width + width, :] = image
    
    return bordered_image


def cropMask(image, mask, offset=0):
    '''
    Return an image which is the cropped mask on a black background 
    with possibly a black fringe (if offset>0)
    '''
    tmp = image.copy()
    tmp = tmp[:,:,::-1]
    tmp[~mask['segmentation']]=0
    x=mask['bbox'][0]
    y=mask['bbox'][1]
    h=mask['bbox'][2]
    w=mask['bbox'][3]
    tmp = tmp[y:y+w,x:x+h,:]
    if offset>0:
        tmp = addBorder(tmp,offset)
    return tmp


def cropMaskNoCrop(image, mask):
    '''
    Return zerowed image which but on the mask
    '''
    tmp = image.copy()
    tmp[~mask['segmentation']]=0
    return tmp


def visualizeSA(image, masks=[],title='',axis='off'):
    '''
    Display colored masks on image
    '''
    
    if type(masks) is not list: masks = [masks]
    
    plt.figure()
    plt.imshow(image)
    plt.title(title)
    show_anns(masks)
    plt.axis(axis)
    plt.show()
    
    
    
def visualizeOnly(image, masks=[],title_='',axis='off'):
    '''
    Display zeroed image everywhere but on the mask(s)
    Masks input can be either a single mask on list of masks
    '''
    
    if type(masks) is not list: masks = [masks]
    
    msk = np.zeros_like(image[:,:,0]).astype(bool)
    for m in masks:
        msk = np.logical_or(msk,m['segmentation'])
    msk = {'segmentation':msk}
    
    msked =cropMaskNoCrop(image,msk)
    plt.figure()
    plt.imshow(msked)
    plt.title(title_)
    plt.axis(axis)
    plt.show()



def matchingAnnotations(anns1:list,anns2:list,th_iou=0.75):
    '''
    Given two lists of dict: anns1 and anns2
    '''
    
    # get all heads
    are_overlapping = maskOverlay2(anns2)
    heads = [i[0] for i in are_overlapping]
    
    # init container
    best_c_id = -1*np.ones(len(anns1))
    best_ious = best_c_id.copy()
    
    for idx, ann1 in enumerate(anns1):
        # which list of indices from are_overlapping intersects the ann2 mask
        overlaying_heads = [head_idx for head_idx,_ in enumerate(are_overlapping) if np.logical_and(ann1['segmentation'],anns2[are_overlapping[head_idx][0]]['segmentation'],).any()]         
        
        # in these lists, which index corresponds to the best matching candidate
        for h_idx in overlaying_heads:
            head_members = are_overlapping[h_idx]
            
            for candidate in head_members:
                ann2 = anns2[candidate]
                if np.logical_and(ann1['segmentation'],ann2['segmentation']).any():
                    iou = IoU(ann1,ann2)
                    if iou > th_iou and iou > best_ious[idx]:
                        best_c_id[idx] = ann2['id']
                        best_ious[idx] = iou
                    
    is_a_match = []
    ious = []
    missed = []
    for idx,ann1 in enumerate(anns1):
        if best_c_id[idx]<0: missed.append(ann1['id'])
        else:
            is_a_match.append([ann1['id'],int(best_c_id[idx])])
            ious.append(best_ious[idx])

    return is_a_match,ious,missed



def replaceAnnotations(anns1,anns2):
    '''
    replace 'segmentation', 'bbox, and 'area' fields from anns1
    with 'segmentation', 'bbox, and 'area' fields from anns2
    '''
    modified = deepcopy(anns1)
    is_a_match = matchingAnnotations(anns1,anns2)
    #print(is_a_match)
    for pair in is_a_match:
        modified[pair[0]]['segmentation']= anns2[pair[1]]['segmentation']
        modified[pair[0]]['bbox']= anns2[pair[1]]['bbox']
        modified[pair[0]]['area']= anns2[pair[1]]['area']
    return modified


def loadDecodedAnnotations(src,image_name='all'):
    if type(src) == list:
        if image_name == 'all': return 'please enter an image_name'
        else: return decodeMasks(readJson([p for p in src if image_name.replace('.jpg','_SA_masks.json') in p][0]))
        
    elif type(src) == str:
        if image_name == 'all': return decodeMasks(readJson(src)['annotations'])
        else: return [i for i in decodeMasks(readJson(src)['annotations']) if i['image_id']==int(image_name.split('.')[0])]