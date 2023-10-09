from config_.paths import *

from helper.lib_SA import show_anns,initSam
from helper.paths import collectPaths,getImagePath
from helper.tuning import imreadRGB,readJson,writeJson
from helper.annotations import correctMaskAoI
from helper.common_libraries import cv2,random,os,join,exists,plt

from segment_anything import SamAutomaticMaskGenerator

#**************************************************************************************************************
getRandomElement = lambda collection: random.sample(collection,1)[0]
display_y = 500
_duo_ = list()

#********************************** USER INPUT REQUESTS *********************************************
def requestBool(display_text):
    """
    Loops until the user presses either 'y' or 'n'.
    Returns True if 'y' was pressed, False for 'n' 
    """
    while True:
        request = input(display_text+' [y/n]: ')
        if request == 'n': request = False
        elif request == 'y': request = True
        if type(request)==bool:break
    return request


def requestInt(display_text, a=0,b=0):
    """
    Loops until the user enters an int value within the requested interval
    """
    a = max(0,a)
    b = max(0,b)
    while True:
        # no min, no max
        if a ==0 and b==0:
            request = int(input(f'{display_text}: '))
            if a <= request: break 
        # min, no max
        elif a>0 and b ==0:
            request = int(input(f'{display_text}, [{a},Inf[: '))
            if a <= request: break 
        # no min,max
        elif a==0 and b>0:
            request = int(input(f'{display_text}, [0,{b}]: '))
            if a <= request and request<=b: break 
        # range
        else:
            request = int(input(f'{display_text}, [{a},{b}]: '))
            if a <= request and request<=b: break 
    return request


def requestFloat(display_text, a_included=True,b_included=True):
    """
    Loops until the user enters a float value within the requested interval
    """
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
    """
    Loops until the user enters an existing channel directory rooted at {path_root}
    Returns the entered value as well as the updated path
    """
    while True:
        channel = input("Channel to process: ")
        if exists(join(path_root, channel)):break
    return channel, join(path_root,channel)


def requestPtsPerSide(path_root=path_root_masks_from_images,as_str=True):
    """
    Loops until the user enters an existing "points directory" rooted at {path_root}
    Returns the entered value as well as the updated path
    """
    while True:
        points_per_side = requestInt('SA masks processeed with _ points per side')
        path = join(path_root, str(points_per_side))
        if exists(path): break 
    return str(points_per_side) if as_str else points_per_side, path


def requestTimestamp(channel):
    """
    Loops until the user enters a timestamp leading to an existing image.
    Returns the timestamp as well as the image path
    """
    while True:
        timestamp = f'{channel}_' + input(f'Timestamp_id: {channel}_')
        path_file_image = getImagePath(timestamp)
        if exists(path_file_image):break
    return timestamp,path_file_image


def augmentCategories(path_file_json):
    """
    Displays the categories existing it the coco_annotations file located at {path_file_json}
    Offers the possibility to directly add new categories
    Returns the updated coco file
    """
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
        
        # request to add new categories
        if nb_categories==0 or requestBool("Create new category"): 
            # check if the category name,id are not already existing
            while True:
                name = input("New category name: ")
                if name not in  [item['name'] for item in available_categories]:break
            while True:
                cat = requestInt("New category id")
                if cat not in [item['id'] for item in available_categories]:break
            # add new category to coco files, write changes
            coco_annotations['categories'].append({
                "supercategory":cat,
                "id":cat,
                "name":input("New category name: ")})
            writeJson(path_file_json,coco_annotations)
            print('Added !')
            print('')
        else: break
    # return updated coco file
    return coco_annotations


def requestSrcPath():
    """
    Loops until the user provides an existing path to a folder
    """
    while True:
        path_dir_src = input('Path to source directory: ')
        if exists(path_dir_src):break
    return path_dir_src


def requestDate(start=True):
    """
    Loops until the user provides a date matching the requested format
    """
    while True:
        print('Please follow requested format: yyyy-mm-dd')
        user_input = input('From: ' if start else 'To: ')
        if len(user_input)==10 and user_input[4]=='-' and user_input[7]=='-':
            tmp = user_input.split('-')
            year = int(tmp[0])
            month = int(tmp[1])
            day = int(tmp[2])
            break
    return year, month, day


def pruneHead(path_list,year,month,day,included=True):
    """
    Removes all paths  containing a timestamp earlier than (year month day) from the path list
    """
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
    """
    Removes all paths  containing a timestamp later than (year month day) from the path list
    """
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
    """
    Removes all paths out of the range [start_date, stop_date] from the path_list
    """
    start_year, start_month, start_day = requestDate()
    stop_year,stop_month,stop_day = requestDate(start=False)
    
    path_list = pruneHead(path_list,start_year,start_month,start_day)
    path_list = pruneTail(path_list,stop_year,stop_month,stop_day)
    
    return path_list


def pruneListException(path_list):
    """
    Removes all paths within the range [start_date, stop_date] from the path_list
    """
    start_year, start_month, start_day = requestDate()
    stop_year,stop_month,stop_day = requestDate(start=False)
    
    head_chunk = pruneTail(path_list,start_year,start_month,start_day,included=False)
    tail_chunk = pruneHead(path_list,stop_year,stop_month,stop_day,included=False)
    
    return head_chunk+tail_chunk


def pruneList(path_list):
    """
    c.f. employed functions
    """
    if requestBool('Interval'): path_list = pruneListFromTo(path_list)
    if requestBool('Except'): path_list = pruneListException(path_list)
    return path_list


def getPathList(path_root):
    """
    c.f. employed functions
    """
    print('Collecting files...')
    path_list = collectPaths(path_root)
    path_list = pruneList(path_list)
    return path_list


# ************************************* DISPLAYS OUTPUT **********************************
def imagePreview(image, masks=[], display_y=display_y):
    """
    Displays the {image} and wait for validation/rejection [y/n] together with some {masks}, if provided
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
    
    # scale before to display
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
    Displays the {image} and wait for twos clicks
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
    """
    Displays an image chosen from path_list_images and displays it
    The user can provide two input points, which define the AoI vertical limits
    """
    key = 'n'
    while True:
        # get random image to display
        image = imreadRGB(getRandomElement(path_list_images))
        if key == 'y': break
        # get points defining AoI
        print("Define an area of interest: ")
        duo = getDuo(image)
        # hide AoI and confirm background area
        print("Validate background area [y/n]: ")
        image[:,duo[0]:duo[1],:] = 0 
        key = imagePreview(image)
        if key == 'n': duo.clear()
    return duo


def setGenerator(path_list_images,duo=-1):
    """
    Initializes a SAM automatic mask generator using {points_per_side} points to segment the image
    The points will be processed per batches of {points_per_batch}
    Displays a preview of the segmentation density and wait for validation
    Returns the activated and validated generator
    """
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
            #box_nms_thresh = 0.75,
            #min_mask_region_area = 2000
            )
        
        # get random image to display and compute masks
        image = imreadRGB(getRandomElement(path_list_images))
        processed_image = image if duo==-1 else image[:,duo[0]:duo[1],:]
        masks = mask_generator.generate(processed_image)
        if duo!=-1: 
            for mask in masks:
                mask = correctMaskAoI(mask,duo,image.shape[:2])
        # display result and ask for validation
        print("Validate mask density [y/n]: ")
        key = imagePreview(image,masks)
    return mask_generator,points_per_side