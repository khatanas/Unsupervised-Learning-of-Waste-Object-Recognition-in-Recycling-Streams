from config_.paths import path_root_GT

from helper.tuning import readJson, writeJson,imreadRGB
from helper.paths import getImagePath,getId,getChannel,getName
from helper.annotations import encodeMasks

from helper.common_libraries import random,join,cv2,np,os

#**************************************************************************************************************
getPathListFromCoco = lambda coco_annotations:[getImagePath(getId(item['file_name'])) for item in coco_annotations['images']]
getCocoName =  lambda path: getImagePath(getId(path)).split(f'{getChannel(path)}/')[-1]
getCocoImageId = lambda path_file_any,coco_annotations: [item['id'] for item in coco_annotations['images'] if item['file_name']==getCocoName(path_file_any)][0]
getAnnsFromImageId = lambda image_id, coco_annotations: [item for item in coco_annotations['annotations'] if item['image_id']==image_id]


def masksFromPolygons(coco_annotations):
    """
    Converts Polygon format to RLE format
    """
    images = coco_annotations['images']
    anns = coco_annotations['annotations']
    
    # quick_dict: fast search of annotations corresponding to image_id
    quick_dict = {item['id']:[] for item in images}
    for k,ann in enumerate(anns): quick_dict[ann['image_id']].append(k)
    
    # iterate over all images (need image dimension)
    for k,image in enumerate(images):
        if k %100==0:print(f'{k}/{len(images)}')
        h = image['height']
        w = image['width']
        image_id = image['id']
        
        # iterate over all ann of image
        for k in quick_dict[image_id]:
            ann = anns[k]
            # just to be really sure
            if ann['image_id'] == image_id and type(ann['segmentation'])==list:
                mask = np.zeros([h,w],dtype=np.uint8)
                
                polygons = ann['segmentation']
                
                # iterate over all polygons of ann (disconnected mask)
                for polygon in polygons:
                    pts = np.array(polygon, dtype=np.int32).reshape((-1, 2))
                    cv2.fillPoly(mask, [pts], 1)
                
                # collect masks
                ann['segmentation'] = mask
                anns[k] = encodeMasks([ann])[0]
            else:print('issue')
            
    return coco_annotations


def initCocoImage(path_file_image,k):
    """
    Returns an image coco dictionary for the image located at {path_file_image}
    """
    image = imreadRGB(path_file_image)
    height,width = image.shape[:2]
    image_dict = {
        "file_name": getCocoName(path_file_image),
        "height":height,
        "width":width,
        "id":k}
    return image_dict


def initCocoAnnotations(path_file_json,path_list_images=[],categories=[],annotations=[]):
    """
    Initializes a coco file
    """
    images = []
    for k,path in enumerate(path_list_images): images.append(initCocoImage(path,k))
    coco_annotations = {
            "info":{},
            "licenses":[],
            "images":images,
            "categories":categories,
            "annotations":annotations
    }
    writeJson(path_file_json,coco_annotations)
    return coco_annotations


def makeIdUnique(path_file_json,coco_annotations=False):
    """
    Makes the ids of the annotations unique
    """
    if not coco_annotations: coco_annotations=readJson(path_file_json)
    anns = coco_annotations['annotations']
    for k,ann in enumerate(anns): ann['id'] = k
    coco_annotations['annotations']=anns
    writeJson(path_file_json,coco_annotations)
    return coco_annotations


def clearAnnotations(path_file_json):
    '''
    Empties the "annotation" field of json file located at path_file_json
    '''
    coco_annotations = readJson(path_file_json)
    coco_annotations['annotations'] = []
    writeJson(path_file_json,coco_annotations)


def splitCoco(path_file_json, path_dir_output, nb=0, file_names=[]):
    '''
    Performs train-test split.
    if test_input is an int, create a te partition of len == test_input
    if test_input is a list, it has to be a list of existing file_name in the original coco_annotations. The te partitions contain the annotations related to those images. 
    '''
    coco_annotations = readJson(path_file_json)
    json_name = getName(path_file_json).split('.')[0]
    
    if file_names: test_list = file_names
    elif nb:
        random.seed(10)
        test_list = sorted(random.sample(sorted([d['file_name'] for d in coco_annotations['images']]),nb))
    
    lod_images_te = [d for d in coco_annotations['images'] if d['file_name'] in test_list]
    lod_images_tr = [d for d in coco_annotations['images'] if d['file_name'] not in test_list]
    
    list_image_id_te = [d['id'] for d in lod_images_te]
    list_image_id_tr = [d['id'] for d in lod_images_tr]
    
    lod_annotations_te = [d for d in coco_annotations['annotations'] if d['image_id'] in list_image_id_te]
    lod_annotations_tr = [d for d in coco_annotations['annotations'] if d['image_id'] in list_image_id_tr]
    
    coco_dict_te = {
        'info': coco_annotations['info'],
        'licenses': coco_annotations['licenses'],
        'images': lod_images_te,
        'categories': coco_annotations['categories'],
        'annotations': lod_annotations_te
        }
    
    coco_dict_tr = {
        'info': coco_annotations['info'],
        'licenses': coco_annotations['licenses'],
        'images': lod_images_tr,
        'categories': coco_annotations['categories'],
        'annotations': lod_annotations_tr
        }
    
    writeJson(join(path_dir_output,f'{json_name}_te.json'),coco_dict_te)
    writeJson(join(path_dir_output,f'{json_name}_tr.json'),coco_dict_tr)