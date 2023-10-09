from config_.parameters import th_min_area,th_iou_duplicates
from config_.paths import *

from helper.tuning import imreadRGB,flattenList
from helper.paths import getImagePath
from helper.common_libraries import np,plt,deepcopy,join,maskUtils,label

#**************************************************************************************************************
updateArea = lambda binary: int(binary.sum())
lookupTable = lambda masks: {mask['id']:m_th for m_th,mask in enumerate(masks)}
bboxArea = lambda mask: [mask['bbox'][2]*mask['bbox'][3]]


def encodeMasks(masks):
    """
    Binary to RLE
    """
    encoded = deepcopy(masks)
    for mask, item in zip(masks,encoded):
        binary_mask = mask['segmentation']
        rle = maskUtils.encode(np.asfortranarray(binary_mask))
        rle['counts'] = rle['counts'].decode('utf-8')
        item['segmentation'] = rle
    return encoded


def decodeMasks(masks):
    """
    RLE to binary
    """
    decoded = deepcopy(masks)
    for mask, item in zip(masks,decoded):
        rle = mask['segmentation']
        #rle['counts']= rle['counts'].encode('utf-8')
        item['segmentation'] = maskUtils.decode(rle).astype(bool)
    return decoded


def updateBbox(binary):
    """
    Returns the bbox_xywh of the input mask
    """
    rle = maskUtils.encode(np.asfortranarray(binary.astype(np.uint8)))
    bbox = maskUtils.toBbox(rle)
    return [int(b) for b in bbox]


def bboxLocation(mask):
    """
    Converts bbox_xywh format to [[x1,x2],[y1,y2]]
    """
    bbox_x_start = mask['bbox'][0]
    bbox_x_stop = bbox_x_start + mask['bbox'][2]
    bbox_y_start = mask['bbox'][1]
    bbox_y_stop = bbox_y_start + mask['bbox'][3]
    
    return([[bbox_x_start,bbox_x_stop],[bbox_y_start,bbox_y_stop]])


def IoU(mask1,mask2):
    """
    Computes the IoU between mask1 and mask2
    """
    binary1 = mask1['segmentation']
    binary2 = mask2['segmentation']
    
    inter = np.logical_and(binary1,binary2).sum()
    union = np.logical_or(binary1,binary2).sum()
    return inter/union


def filterMasks(masks, id_list):
    """
    Returns the mask having their id in the id_list
    """
    token = False
    if type(id_list)==int: 
        token = True
        id_list = [id_list]
    filtered = [mask for mask in masks if any([mask['id']==id for id in id_list])]
    return filtered[0] if token else filtered


def maskOverlap(masks, th_IoU=0.015):
    '''
    Returns a nested list of overlapping masks, sorted by area.
    Each list contains the mask[id] of the masks overlapping with each other.
    '''
    # put the biggest annotations at the beginning
    sorted_annotations = sorted(masks, key=lambda mask: mask['area'],reverse=True)
    
    are_overlapping = []
    to_skip = []
    # iterate over every masks once
    for ann1 in sorted_annotations:
        if ann1['id'] in to_skip: continue
        binary1 = ann1['segmentation']
        # init new overlapping list, and add node to skipping list (speedup)
        overlapping_list = [ann1['id']]
        to_skip.append(ann1['id'])
        
        # look for any intersecting masks
        for ann2 in sorted_annotations:
            if ann2['id'] in to_skip: continue
            binary2 = ann2['segmentation']
            if np.logical_and(binary1 , binary2).any():
                # the IoU is used to filter segmentation noise. 
                # Sometimes, some mask are said to be overlapping because of only a few pixels.
                if IoU(ann1,ann2)>th_IoU: 
                    # add overlapping masks to list of overlapping, and add node to skipping list (speedup)
                    overlapping_list.append(ann2['id'])
                    to_skip.append(ann2['id'])
        are_overlapping.append(overlapping_list) 
    return are_overlapping


def findDuplicates(masks,th_duplicate=th_iou_duplicates):
    """
    Returns matching annotations under the form: 
    [
        [self_id_0 , [match_id_0, ... , match_id_i],
        ...
        [self_id_k , [match_id_1, ... , match_id_j],
    ]
    with k < len(masks)
    """
    # mask['id'] ==> m_th mask 
    lookup = lookupTable(masks)
    
    # init candidate_id and to_skip container
    similar_to = [[] for _ in masks]
    to_skip = []    
    
    # get overlapping lists of masks
    are_overlapping = maskOverlap(masks)
    for k_th,mask in enumerate(masks):
        
        # if mask already in a list, don't process
        if mask['id'] in to_skip: continue
        
        # get all sublists containing current mask from are_overlapping
        matching_sublists = [sublist for sublist in are_overlapping if mask['id'] in sublist]
        
        # gather all matching masks in a single list 
        matching_ids = flattenList(matching_sublists)
        
        # find matching annotations with IoU>th_duplicate
        for matching_id in matching_ids:
            if (matching_id==mask['id']) or (matching_id in to_skip): continue
            similar_mask = masks[lookup[matching_id]]
            if np.logical_and(mask['segmentation'],similar_mask['segmentation']).any():
                if IoU(mask,similar_mask)>th_duplicate:
                    similar_to[k_th].append(similar_mask['id'])
                    to_skip.append(similar_mask['id'])
            
    return [[mask['id'], similar_to[k_th]] for k_th,mask in enumerate(masks) if similar_to[k_th]]


def cleanDuplicates(masks, th_duplicate=th_iou_duplicates):
    """
    Cleans off the duplicated masks by merging them with their "original" mask
    """
    # get pairs of duplicates: [[id, list_of_ids],....]
    lists_of_duplicates = findDuplicates(masks,th_duplicate=th_duplicate)
    
    # gather all ids in any list_of_ids (the duplicates of a mask)
    to_be_removed_id = flattenList([pair[1] for pair in lists_of_duplicates])
    
    # for each pair
    for pair in lists_of_duplicates:
        # get "the original" (it has a better SAM stability score)
        mask = masks[pair[0]]
        # and do an "OR" operation with its duplicates (aribtrary, we could also just get rid of them)
        for id1 in pair[1]: mask['segmentation']=np.logical_or(mask['segmentation'],masks[id1]['segmentation'])
        mask['area'] = updateArea(mask['segmentation'])
        mask['bbox'] = updateBbox(mask['segmentation'])
    # remove duplicates from masks list
    masks = [mask for mask in masks if mask['id'] not in to_be_removed_id]
    # update the ids
    for idx,mask in enumerate(masks): mask['id'] = idx
    return masks


def cleanMasks(masks, th_area=th_min_area, th_duplicate=th_iou_duplicates):
    """
    Cleans masks areas smaller than th_area (pixels), and removes the duplicated mask having an IoU>th_duplicate
    """
    for mask in masks:
        if 'id' in mask.keys():continue
        token=False
        
        # count separate items
        tmp = label(mask['segmentation'])
        # if more than one
        if tmp.max()>1:
            # check area and remove if #pixels < threshold
            for i in range(1, len(np.unique(tmp))):
                if (tmp==i).sum()<th_area: 
                    mask['segmentation'][tmp==i]=0
                    token = True
            # once all small areas are removed, update area and bbox
            if token:
                mask['area'] = updateArea(mask['segmentation'])
                mask['bbox'] = updateBbox(mask['segmentation'])
    # remove masks having their final overall area <= th_area
    masks = [mask for mask in masks if mask['area']>=th_area]
    
    # assign an id to each mask
    for idx,mask in enumerate(masks): mask['id'] = idx
    
    # filters off duplicates
    masks = cleanDuplicates(masks, th_duplicate=th_duplicate)
    
    return masks


def addBorder(image, border_width,value):
    '''
    Adds a border of {border_width} around the image
    '''
    # RGB
    if len(image.shape)==3:
        height, width, channels = image.shape
        new_height = height + 2 * border_width
        new_width = width + 2 * border_width
        bordered_image = value*np.ones((new_height, new_width, channels), dtype=image.dtype)
        bordered_image[border_width:border_width + height, border_width:border_width + width,:] = image
    # Grey
    else:
        height, width = image.shape
        new_height = height + 2 * border_width
        new_width = width + 2 * border_width
        bordered_image = value*np.ones((new_height, new_width), dtype=image.dtype)
        bordered_image[border_width:border_width + height, border_width:border_width + width] = image
    
    return bordered_image


def cropMask(image, mask, offset=0,value=0):
    '''
    Returns an image cropped using the bbox, where the pixels outside the segmentation mask are set to {value}.
    It is possible to add a black fringe if {offset}>0.
    '''
    cropped = image.copy() if type(image) is np.ndarray else imreadRGB(getImagePath(mask['timestamp_id']))
    cropped[~mask['segmentation']]=value
    x=int(mask['bbox'][0])
    y=int(mask['bbox'][1])
    h=int(mask['bbox'][2])
    w=int(mask['bbox'][3])
    cropped = cropped[y:y+w,x:x+h,:] if len(cropped.shape)==3 else cropped[y:y+w,x:x+h] 
    if offset>0: cropped = addBorder(cropped,offset,value)
    return cropped


def matchingAnnotations(anns1:list,anns2:list,th_iou=0.75):
    """
    Given anns1, anns2, it will return three lists
    i) a list of pairs [[id_from_ann1, id_from_ann2],...]
    ii) a list of IoU score [[0.9876], ...]
    iii) a list of unmatched ids from anns1. 
    """
    
    # id ==> m-th mask
    lookup = lookupTable(anns2)
    
    # get all heads
    are_overlapping = maskOverlap(anns2)
    head_ids = [i[0] for i in are_overlapping]
    
    # init container
    best_c_id = [-1 for _ in anns1]
    best_iou = [-1 for _ in anns1]
    
    for idx, ann1 in enumerate(anns1):
        # which list of indices from are_overlapping intersects the ann2 mask
        overlapping_heads = []
        binary1 = ann1['segmentation']
        for k_th,id in enumerate(head_ids):
            binary2 = anns2[lookup[id]]['segmentation']
            if np.logical_and(binary1,binary2).any(): overlapping_heads.append(k_th)
        
        # in these lists, which index corresponds to the best matching candidate
        for k_th in overlapping_heads:
            head_members = are_overlapping[k_th]
            
            for member_id in head_members:
                ann2 = anns2[lookup[member_id]]
                if np.logical_and(binary1,ann2['segmentation']).any():
                    iou = IoU(ann1,ann2)
                    if iou > th_iou and iou > best_iou[idx]:
                        best_c_id[idx] = ann2['id']
                        best_iou[idx] = iou
    
    is_a_match = [[ann1['id'], match] for ann1,match in zip(anns1,best_c_id) if match!=-1]
    ious = [iou for iou in best_iou if iou!=-1]
    missed = [ann1['id'] for ann1,match in zip(anns1,best_c_id) if match==-1]
        
    return is_a_match,ious,missed


def replaceAnnotations(anns1,anns2, th_iou=0.75):
    '''
    replace 'segmentation', 'bbox, and 'area' fields from anns1
    with 'segmentation', 'bbox, and 'area' fields from anns2
    '''
    modified = deepcopy(anns1)
    is_a_match = matchingAnnotations(anns1,anns2,th_iou=th_iou)
    
    lookup1 = {item['id']:k for k,item in enumerate(anns1)}
    lookup2 = {item['id']:k for k,item in enumerate(anns2)}
    for pair in is_a_match[0]:
        to_modify = modified[lookup1[pair[0]]]
        
        tmp_loc = lookup2[pair[1]]
        to_modify['segmentation'] = anns2[tmp_loc]['segmentation']
        to_modify['bbox'] = anns2[tmp_loc]['bbox']
        to_modify['area'] = anns2[tmp_loc]['area']
        modified[modified.index(to_modify)]=to_modify
        
    return modified


def getHeads(masks):
    """
    Returns all masks classified as head
    """
    are_overlapping = maskOverlap(masks)
    heads_id = [sublist[0] for sublist in are_overlapping]
    filtered = filterMasks(masks,heads_id)
    return filtered


def getAoIs(masks):
    """
    Returns all masks belonging to the AoI or crossing it
    """
    in_AoI = [mask['id'] for mask in masks if mask['AoI']>=0]
    return filterMasks(masks,in_AoI)


def visualizeOnly(image, masks=[], value=0, title='',show_title=True, axis='off', path_save=False):
    """
    Displays the area of the {image} covered by the {masks}. The rest of the image is set to {value}
    """
    # convert single mask to a list 
    if type(masks) is not list: masks = [masks]
    
    # if the image input is an empty list, the image is read using the timestamp id embedded in the first mask of the list
    masked = imreadRGB(getImagePath(masks[0]['timestamp_id'])) if type(image)==list else image.copy()
    
    # sum the masks and mask the image
    mask_acc = np.zeros_like(masked[:,:,0]).astype(bool)
    for mask in masks: mask_acc = np.logical_or(mask_acc,mask['segmentation'])
    if masks: masked[~mask_acc] = value
    
    # plot
    plt.figure()
    plt.imshow(masked)
    if show_title: plt.title(title)
    plt.axis(axis)
    if path_save!=False: plt.savefig(join(path_save,f'{title}.jpg'),bbox_inches='tight')
    plt.show()


def padBinary(binary_mask,duo,hw0):
    """
    Adds bands of "False" values to the left and the right of the binary mask
    The returned masks has the width specified by {hw0[1]}
    """
    pad_left = duo[0]
    pad_right = hw0[1]-duo[1]
    padded = np.pad(binary_mask,((0,0),(pad_left,pad_right)),mode='constant',constant_values=False)
    assert(padded.shape[:2])==hw0
    return padded


def correctMaskAoI(mask,duo,hw0):
    """
    Corrects the the listed values of the mask and returns the updated mask
    """
    mask['segmentation'] = padBinary(mask['segmentation'],duo,hw0)
    mask['size'] = hw0
    mask['bbox'][0] += duo[0]
    mask['point_coords'][0][0] += duo[0]
    mask['crop_box'][2] = hw0[1]
    return mask