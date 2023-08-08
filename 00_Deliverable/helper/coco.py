from helper.tuning import readJson, writeJson,imreadRGB
from helper.paths import getName,getImagePath,getId,getChannel

from helper.common_libraries import random,shuffle,listdir,makedirs,join


getPathListFromCoco = lambda json_file:[getImagePath(getId(item['file_name'])) for item in json_file['images']]
getCocoName =  lambda path: getImagePath(getId(path)).split(f'{getChannel(path)}/')[-1]
getCocoImageId = lambda path,json_file: [item['id'] for item in json_file['images'] if item['file_name']==getCocoName(path)][0]


def initCocoAnnotations(path_file_json,path_list_images,categories=[]):
    channel = getName(path_list_images[0])[:7]
    images = []
    for k,path in enumerate(path_list_images):
        image = imreadRGB(path)
        height,width = image.shape[:2]
        images.append(
            {
                "file_name": path.split(f'{channel}/')[-1],
                "height":height,
                "width":width,
                "id":k
            }
        )
    json_file = {
            "info":{},
            "licenses":[],
            "images":images,
            "categories":categories,
            "annotations":[]
    }
    writeJson(path_file_json,json_file)
    return json_file


def makeIdUnique(path_file_json,json_file=False):
    if not json_file: json_file=readJson(path_file_json)
    anns = json_file['annotations']
    for k,ann in enumerate(anns): ann['id'] = k
    json_file['annotations']=anns
    writeJson(path_file_json,json_file)
    return json_file


def addCategory(path_file_json,json_file=False):
    if not json_file: json_file=readJson(path_file_json)
    nb_categories = len(json_file['categories'])
    cat = int('9'+str(nb_categories+1).zfill(4))
    json_file['categories'].append({
        "supercategory":cat,
        "id":cat,
        "name":input("New category name: ")}
    )
    writeJson(path_file_json,json_file)
    print('Added !')
    print('')
    return readJson(path_file_json)


def clearAnnotations(path_file_json):
    '''
    Empty the "annotation" field of json file located at path_file_json
    '''
    json_file = readJson(path_file_json)
    json_file['annotations'] = []
    writeJson(path_file_json,json_file)


def trainTestSplit(path_file_json, test_input):
    '''
    Perform train-test split.
    if test_input is an int, create a te partition of len == test_input
    if test_input is a list, it has to be a list of existing file_name in the original json_file. The te partitions contain the annotations related to those images. 
    '''
    json_file = readJson(path_file_json)
    json_name = path_file_json.split('/')[-1]
    path_dir_split = path_file_json.replace(f'/{json_name}','')
    
    if type(test_input)==list: test_list = test_input
    elif type(test_input)==int:
        image_list = sorted([d['file_name'] for d in json_file['images']])
        random.seed(10)
        shuffle(image_list)
        test_list = image_list[test_input]
        test_list = sorted(test_list)
    
    lod_images_te = [d for d in json_file['images'] if d['file_name'] in test_list]
    lod_images_tr = [d for d in json_file['images'] if d['file_name'] not in test_list]

    list_image_id_te = [d['id'] for d in lod_images_te]
    list_image_id_tr = [d['id'] for d in lod_images_tr]

    lod_annotations_te = [d for d in json_file['annotations'] if d['image_id'] in list_image_id_te]
    lod_annotations_tr = [d for d in json_file['annotations'] if d['image_id'] in list_image_id_tr]

    coco_dict_te = {
        'info': json_file['info'],
        'licenses': json_file['licenses'],
        'images': lod_images_te,
        'categories': json_file['categories'],
        'annotations': lod_annotations_te
        }
    
    coco_dict_tr = {
        'info': json_file['info'],
        'licenses': json_file['licenses'],
        'images': lod_images_tr,
        'categories': json_file['categories'],
        'annotations': lod_annotations_tr
        }
    
    writeJson(join(path_dir_split,f'te_{json_name}'),coco_dict_te)
    writeJson(join(path_dir_split,f'tr_{json_name}'),coco_dict_tr)