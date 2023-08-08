from config.paths import *
from config.classification import getTaxonomy,getSuperCategories

from helper.lib_SA import show_anns,initSam
from helper.paths import collectPaths
from helper.tuning import imreadRGB,readJson
from helper.coco import addCategory
from helper.common_libraries import cv2,random,os,join,exists,plt

from segment_anything import SamAutomaticMaskGenerator

getRandomElement = lambda collection: random.sample(collection,1)[0]
#********************************** USER INPUT REQUESTS *********************************************
def requestBool(display_text):
    while True:
        request = input(display_text+' [y/n]: ')
        if request == 'n': request = False
        elif request == 'y': request = True
        if type(request)==bool:break
    return request


def requestInt(display_text,a=0,b=0):
    a = max(0,a)
    b = max(0,b)
    while True:
        # no min, no max
        if a ==0 and b==0:
            request = int(input(f'{display_text}: '))
            if a <= request: break 
        # min, no max
        elif a>0 and b ==0:
            request = int(input(f'{display_text}, >={a}: '))
            if a <= request: break 
        # no min,max
        elif a==0 and b>0:
            request = int(input(f'{display_text}, <={b}: '))
            if a <= request and request<=b: break 
        # range
        else:
            request = int(input(f'{display_text}, [{a},{b}]: '))
            if a <= request and request<=b: break 
    return request


def requestFloat(display_text,a_included = True,b_included=True):
    while True:
        if a_included and b_included:
            request = float(input(f'{display_text}, [0,1]: '))
            if 0<=request and request<=1: break 
        elif a_included and not b_included:
            request = float(input(f'{display_text}, [0,1[: '))
            if 0<=request and request<1:break 
        elif not a_included and b_included:
            request = float(input(f'{display_text}, ]0,1]: '))
            if 0<request and request<=1:break 
        else:
            request = float(input(f'{display_text}, ]0,1[: '))
            if 0<request and request<1:break  
    return request


def requestChannel(path_root):
    while True:
        channel = input("Channel to process: ")
        if exists(join(path_root, channel)):break
    return channel, join(path_root,channel)


def requestPtsPerSide(path_root=path_root_masks_from_images,as_str=True):
    while True:
        points_per_side = requestInt('SA masks processeed with _ points per side')
        path = join(path_root, str(points_per_side))
        if exists(path): break
    return str(points_per_side) if as_str else points_per_side, path


def displayCategories(path_file_json):
    coco_annotations = readJson(path_file_json)
    while True:
        available_categories = coco_annotations['categories']
        nb_categories = len(available_categories)
        
        # display available categories
        if nb_categories == 0: print('\nNo available category')
        else:
            print('\nAvailable categories:')
            for k,cat in enumerate(available_categories):
                print(f' {cat["id"]}: {cat["name"]}')
        
        if nb_categories==0 or requestBool("Create new category"): 
            coco_annotations = addCategory(path_file_json,coco_annotations)
        else: break
        
    return coco_annotations

'''def requestRange(path_root, as_str=True):
    while True:
        nb = requestInt('Minimum nb of objects within taxonomy')
        if exists(join(path_root, f'range_of_{nb}.json')): break
    return str(nb) if as_str else nb, join(path_root, f'range_of_{nb}.json')


def requestThreshold(path_root, as_str=True):
    while True:
        classification_threshold = requestFloat('Classification threshold')
        path = join(path_root, f'{classification_threshold}')
        if exists(path):break
    return str(classification_threshold) if as_str else classification_threshold, path'''


def requestObject(path_root, channel, threshold):
    _,_,names_all,names_subparts = getTaxonomy(channel)
    main_elements = [name for name in names_all if name not in names_subparts]
    while True:
        for k, item in enumerate(main_elements):
            print(f'{k}: {item}')
        looking_for = int(input('Looking for: '))
        if 0 <= looking_for and looking_for < len(main_elements): break
    filled_name = main_elements[looking_for].replace(' ','_')
    return join(path_root, f'{filled_name}_{threshold}.json')


def requestCategory(channel):
    while True:
        for item in getSuperCategories(channel):
            print(f'{item[0]}: {item[1][0]} - {item[1][1]}')
        annotated = int(input('To be annotated: '))
        if annotated in [item[0] for item in getSuperCategories(channel)]: break
    return annotated


def requestSrcPath():
    while True:
        path_dir_src = input('Path to source directory: ')
        if exists(path_dir_src):break
    return path_dir_src


def requestStart():
    while True:
        print('Please follow requested format: yyyy-mm-dd')
        start_time = input('From: ')
        if len(start_time)==10:
            if start_time[4]=='-':
                tmp = start_time.split('-')
                start_year = int(tmp[0])
                start_month = int(tmp[1])
                start_day = int(tmp[2])
                break
    return start_year, start_month, start_day


def requestStop():
    while True:
        stop_time = input('To: ')
        
        if len(stop_time)==10:
            if stop_time[4]=='-':
                tmp = stop_time.split('-')
                stop_year = int(tmp[0])
                stop_month = int(tmp[1])
                stop_day = int(tmp[2])
                break
    return stop_year,stop_month,stop_day


def pruneHead(path_list,year,month,day,included=True):
    path_list = [p for p in path_list if(
        int(p.split('/')[-4])>year or(
            int(p.split('/')[-4])==year and(
                int(p.split('/')[-3])>month or (
                    int(p.split('/')[-3])==month and (
                        int(p.split('/')[-2])>=day if included else int(p.split('/')[-2])>day)
                )
            )
        )
    )]
    return path_list


def pruneTail(path_list,year,month,day,included=True):
    path_list = [p for p in path_list if(
        int(p.split('/')[-4])<year or(
            int(p.split('/')[-4])==year and(
                int(p.split('/')[-3])<month or (
                    int(p.split('/')[-3])==month and (
                        int(p.split('/')[-2])<=day if included else int(p.split('/')[-2])<day)
                )
            )
        )
    )]
    return path_list


def pruneListFromTo(path_list):
    start_year, start_month, start_day = requestStart()
    stop_year,stop_month,stop_day = requestStop()
    
    path_list = pruneHead(path_list,start_year,start_month,start_day)
    path_list = pruneTail(path_list,stop_year,stop_month,stop_day)
    
    return path_list


def pruneListException(path_list):
    start_year, start_month, start_day = requestStart()
    stop_year,stop_month,stop_day = requestStop()
    
    head_chunk = pruneTail(path_list,start_year,start_month,start_day,included=False)
    tail_chunk = pruneHead(path_list,stop_year,stop_month,stop_day,included=False)
    
    return head_chunk+tail_chunk


def pruneList(path_list):
    if requestBool('Interval'): path_list = pruneListFromTo(path_list)
    if requestBool('Except'): path_list = pruneListException(path_list)
    return path_list


def getPathList(path_root):
    print('Collecting files...')
    path_list = collectPaths(path_root)
    path_list = pruneList(path_list)
    return path_list

# ************************************* DISPLAY OUTPUT **********************************
display_y = 500

def imagePreview(image, masks=[], display_y=display_y):
    """
    Display image and wait for validation/rejection [y/n]
    It is possible to add masks to the displayed image
    """
    
    # add masks
    if len(masks)>0: 
        path_file_SA_tmp_img = join(path_lib_SA,'tmp_mask.png')
        plt.figure()
        plt.imshow(image)
        plt.axis('off')
        show_anns(masks)
        plt.savefig(path_file_SA_tmp_img, bbox_inches='tight')
        plt.close()
        image = imreadRGB(path_file_SA_tmp_img)
    
    scaling_factor_y = image.shape[0]/display_y
    resized_image = cv2.resize(image, (int(image.shape[1]/scaling_factor_y),display_y))
    
    cv2.namedWindow("Preview")
    cv2.imshow("Preview", resized_image[:,:,::-1])
    
    while True:
        key = cv2.waitKey(0)
        key = chr(key)
        print(key)
        break
    cv2.destroyAllWindows()
    
    if len(masks)>0: os.remove(path_file_SA_tmp_img)
    return key


_duo_ = list()
def mouse_callback(event, x, y, flags, params):
    """
    getDuo helper function
    """
    global _duo_, _scaling_factor_y_

    # left-click event value is 1
    if event == 1:
        x_click = int(float(x)* _scaling_factor_y_)
        if len(_duo_)==0:
            _duo_.append(x_click)
            print(_duo_[0])
            print('Click on the right imaginary y axis... ')
        elif len(_duo_) < 2:
            if x_click > _duo_[0]: 
                _duo_.append(x_click)
                print(_duo_[1])
                print('Press any key...')
            else: print('Click on a valid right imaginary y axis... ')
        else: print('Press any key...')


def getDuo(image, display_y=display_y):
    """
    Display image and wait for clicks
    """
    global _duo_, _scaling_factor_y_
    
    _scaling_factor_y_ = image.shape[0]/display_y
    resized_image = cv2.resize(image,(int(image.shape[1]/ _scaling_factor_y_),display_y))
    
    cv2.namedWindow('Define area of interest')
    print('Click on the left imaginary y axis... ')
    cv2.setMouseCallback('Define area of interest', mouse_callback)
    cv2.imshow('Define area of interest', resized_image[:,:,::-1])
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    return _duo_


def setAoI(path_list_images):
    key = 'n'
    while True:
        # get random image to display
        image = imreadRGB(getRandomElement(path_list_images))
        if key == 'y': break
        # get points defining AoI
        print("Define an area of interest: ")
        duo = getDuo(image)
        
        # hide AoI and confirm backrgroung area
        print("Validate background area [y/n]: ")
        image[:,duo[0]:duo[1],:] = 0 
        key = imagePreview(image)
        if key == 'n': duo.clear()
    return duo


def setGenerator(path_list_images):
    # choose annotation density
    key = 'n'
    while True:
        if key == 'y': break
        points_per_side = requestInt('Points per side', a=1,b=32)
        points_per_batch = requestInt('Points per batch', a=1,b=32)    
        
        print('Please wait during preview generation...')    
        # init auto SAM
        mask_generator = SamAutomaticMaskGenerator(
            model = initSam(),
            points_per_side=points_per_side,
            points_per_batch=points_per_batch,
            )
        
        # get random image to display and compute masks
        image = imreadRGB(getRandomElement(path_list_images))
        masks = mask_generator.generate(image)
        
        # display result and ask for validation
        print("Validate mask density [y/n]: ")
        key = imagePreview(image,masks)
        #key ='y'
    return mask_generator,points_per_side