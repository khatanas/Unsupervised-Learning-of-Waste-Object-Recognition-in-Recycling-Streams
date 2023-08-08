from config.parameters import th_min_area,th_iou_duplicates
from config.paths import *

from helper.tuning import imreadRGB,readJson
from helper.paths import getImagePath
from helper.common_libraries import np,plt,deepcopy,join,maskUtils,label


def encodeMasks(masks):
    tojson = deepcopy(masks)
    for i in range(len(masks)):
        binary_mask = masks[i]['segmentation']
        rle = maskUtils.encode(np.asfortranarray(binary_mask))
        rle['counts'] = rle['counts'].decode('utf-8')
        tojson[i]['segmentation'] = rle
    return tojson


def decodeMasks(masks):
    topython = deepcopy(masks)
    for i in range(len(masks)):
        rle = masks[i]['segmentation']
        #rle['counts']= rle['counts'].encode('utf-8')
        topython[i]['segmentation'] = maskUtils.decode(rle).astype(bool)
    return topython


def updateArea(mask):
    return int(mask.sum())


def updateBbox(mask):
    rle = maskUtils.encode(np.asfortranarray(mask.astype(np.uint8)))
    bbox = maskUtils.toBbox(rle)
    return [int(b) for b in bbox]


def bboxArea(mask):
    '''
    Return the bbox area of an annotation, in pixels
    '''
    return mask['bbox'][2]*mask['bbox'][3]


def bboxLocation(mask):
    bbox_x_start = mask['bbox'][0]
    bbox_x_stop = bbox_x_start + mask['bbox'][2]
    bbox_y_start = mask['bbox'][1]
    bbox_y_stop = bbox_y_start + mask['bbox'][3]
    
    return([[bbox_x_start,bbox_x_stop],[bbox_y_start,bbox_y_stop]])


def filterMasks(masks, id_list):
    token = False
    if type(id_list)==int: 
        token = True
        id_list = [id_list]
    filtered = [mask for mask in masks if mask['id'] in id_list]
    return filtered[0] if token else filtered


def IoU(mask1,mask2):
    segm1 = mask1['segmentation']
    segm2 = mask2['segmentation']
    
    inter = np.logical_and(segm1,segm2).sum()
    union = np.logical_or(segm1,segm2).sum()
    return inter/union


def maskOverlap(masks, th_IoU=0.015):
    '''
    Return a list of list, sorted by area. Contains the idx of the masks overlapping with each other
    '''
    
    # put the biggest annotations at the beginning
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


def findDuplicates(masks,th_duplicate=th_iou_duplicates):
    """
    Return matching annotations under the form: 
    [
        [self_id_0 , [match_id_0, ... , match_id_i],
        ...
        [self_id_k , [match_id_1, ... , match_id_j],
    ]
    
    with k < len(masks
    """
    
    # init candidate_id and pass_list container
    best_c_id = [[] for _ in range(len(masks))]
    pass_list = []    
    
    # get overlapping lists of masks
    are_overlapping = maskOverlap(masks)
    for self_id, mask in enumerate(masks):
        # intern
        assert mask['id']==self_id
        
        # if mask already in a list, don't process
        if self_id in pass_list:continue
        
        # get all sublists containing current mask from are_overlapping
        matching_sublists = [sublist for sublist in are_overlapping if self_id in sublist]
        
        # gather all matching masks in a single list 
        matching_anns_id = [item for sublist in matching_sublists for item in sublist]
        
        # find matching annotations with IoU>th_duplicate
        for matching_id in matching_anns_id:
            if (matching_id==self_id) or (matching_id in pass_list): continue
            mask_bar = masks[matching_id]
            if np.logical_and(mask['segmentation'],mask_bar['segmentation']).any():
                if IoU(mask,mask_bar) > th_duplicate:
                    best_c_id[mask['id']].append(mask_bar['id'])
                    pass_list.append(mask_bar['id'])
                    
    # prepare output
    is_a_match = []
    for self_id, mask in enumerate(masks):
        if len(best_c_id[self_id])>0: 
            is_a_match.append([self_id,[id for id in best_c_id[self_id]]])
            
    return is_a_match


def cleanDuplicates(masks, th_duplicate=th_iou_duplicates):
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


def cleanMasks(masks, th_area=th_min_area, th_duplicate=th_iou_duplicates):
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
                ann['area'] = updateArea(ann['segmentation'])
                ann['bbox'] = updateBbox(ann['segmentation'])
                
    masks = [mask for mask in masks if mask['area']>=th_area]
    for k,mask in enumerate(masks):mask['id']=k
    masks = cleanDuplicates(masks, th_duplicate=th_duplicate)
    return masks


def addBorder(image, border_width,value):
    '''
    Add a border of {border_width} around the image
    '''
    if len(image.shape)==3:
        height, width, channels = image.shape
        new_height = height + 2 * border_width
        new_width = width + 2 * border_width
        bordered_image = value*np.ones((new_height, new_width, channels), dtype=image.dtype)
        bordered_image[border_width:border_width + height, border_width:border_width + width,:] = image
    else:
        height, width = image.shape
        new_height = height + 2 * border_width
        new_width = width + 2 * border_width
        bordered_image = value*np.ones((new_height, new_width), dtype=image.dtype)
        bordered_image[border_width:border_width + height, border_width:border_width + width] = image
    
    return bordered_image


def cropMask(image, mask, offset=0,value=0):
    '''
    Return an image which is the cropped mask on a black background 
    with possibly a black fringe (if offset>0)
    '''
    tmp = image.copy()
    tmp[~mask['segmentation']]=value
    x=int(mask['bbox'][0])
    y=int(mask['bbox'][1])
    h=int(mask['bbox'][2])
    w=int(mask['bbox'][3])
    tmp = tmp[y:y+w,x:x+h,:] if len(image.shape)==3 else tmp[y:y+w,x:x+h] 
    if offset>0:
        tmp = addBorder(tmp,offset,value)
    return tmp


def cropMask2(mask, offset=0,value=0):
    '''
    Return an image which is the cropped mask on a black background 
    with possibly a black fringe (if offset>0)
    '''
    image = imreadRGB(getImagePath(mask['timestamp_id']))
    tmp = image.copy()
    tmp[~mask['segmentation']]=value
    x=int(mask['bbox'][0])
    y=int(mask['bbox'][1])
    h=int(mask['bbox'][2])
    w=int(mask['bbox'][3])
    tmp = tmp[y:y+w,x:x+h,:] if len(image.shape)==3 else tmp[y:y+w,x:x+h] 
    if offset>0:
        tmp = addBorder(tmp,offset,value)
    return tmp


def cropMaskNoCrop(image, mask, value=0):
    """
    Return a zerowed image everywhere but on the mask
    """
    tmp = image.copy()
    tmp[~mask['segmentation']]=value
    return tmp


def visualizeOnly(image, masks=[], value=0, title='',show_title=True, axis='off', path_save=False):
    '''
    Display zeroed image everywhere but on the mask(s)
    Masks input can be either a single mask on list of masks
    '''
    
    if type(masks) is not list: masks = [masks]
    
    msk = np.zeros_like(image[:,:,0]).astype(bool)
    for m in masks:
        msk = np.logical_or(msk,m['segmentation'])
    msk = {'segmentation':msk}
    
    msked =cropMaskNoCrop(image,msk,value=value)
    plt.figure()
    plt.imshow(msked)
    if show_title: plt.title(title)
    plt.axis(axis)
    if path_save!=False: plt.savefig(join(path_save,f'{title}.jpg'),bbox_inches='tight')
    plt.show()


def visualizeOnly2(masks=[], value=0, title='',show_title=True, axis='off', path_save=False):
    '''
    Display zeroed image everywhere but on the mask(s)
    Masks input can be either a single mask on list of masks
    '''
    
    if type(masks) is not list: masks = [masks]
    image = imreadRGB(getImagePath(masks[0]['timestamp_id']))
    
    msk = np.zeros_like(image[:,:,0]).astype(bool)
    for m in masks:
        msk = np.logical_or(msk,m['segmentation'])
    msk = {'segmentation':msk}
    
    msked =cropMaskNoCrop(image,msk,value=value)
    plt.figure()
    plt.imshow(msked)
    if show_title: plt.title(title)
    plt.axis(axis)
    if path_save!=False: plt.savefig(join(path_save,f'{title}.jpg'),bbox_inches='tight')
    plt.show()



def matchingAnnotations(anns1:list,anns2:list,th_iou=0.75):
    '''
    Given two lists of dict: anns1 and anns2
    '''
    
    # get all heads
    are_overlapping = maskOverlap(anns2)
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


def replaceAnnotations(anns1,anns2, th_iou=0.75):
    '''
    replace 'segmentation', 'bbox, and 'area' fields from anns1
    with 'segmentation', 'bbox, and 'area' fields from anns2
    '''
    modified = deepcopy(anns1)
    is_a_match = matchingAnnotations(anns1,anns2,th_iou=th_iou)
    #print(is_a_match)
    for pair in is_a_match[0]:
        to_modify = [ann for ann in modified if ann['id']==pair[0]][0]
        to_modify['segmentation'] = anns2[pair[1]]['segmentation']
        to_modify['bbox'] = anns2[pair[1]]['bbox']
        to_modify['area'] = anns2[pair[1]]['area']
        modified[modified.index(to_modify)]=to_modify
        
    return modified


def getHeads(masks):
    are_overlapping = maskOverlap(masks)
    heads_id = [sublist[0] for sublist in are_overlapping]
    filtered = filterMasks(masks,heads_id)
    return filtered


def getAoIs(masks):
    in_AoI = [mask['id'] for mask in masks if mask['AoI']>=0]
    return filterMasks(masks,in_AoI)