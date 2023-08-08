from config.parameters import display_width

from helper.lib_SA import *
from helper.annotations import encodeMasks,decodeMasks,updateArea,updateBbox,bboxLocation,IoU
from helper.tuning import *
from helper.paths import getId
from helper.common_libraries import random, deepcopy, remove, exists

from segment_anything import SamPredictor

# where to temporarily save displayed image
path_file_SA_tmp_img = join(path_lib_SA,'tmp_mask.png')

def printInstructions():
    print('\n',
    'When the image is displayed:\n',
    'i) Click to add some points (multiple clicks possible):\n',
    '     - left click: add point to include (+)\n',
    '     - right click: add point to avoid (-)\n',
    '     ==> when done with clicking, press any key\n\n',
    
    'ii) The diplayed image shows clicks - available actions:\n',
    '     - press "w" to remove last click\n',
    '     - press "q" to remove all clicks\n',
    '     - press "s" to validate current annotation\n',
    '     - press "f" to pop current annotation from json file\n\n',
    
    'iii) The diplayed image do not show any click - available actions:\n',
    '     - press "c" to cancel previous annotation\n',
    '     - press "d" to save current annotations to json file and move to next image\n',
    '     - press "a" to save current annotations to json file and move to previous image\n',
    '     - press "q" to save current annotations and quit annotation process'
)

def hideFromImage(annotations,image,a=25,b=230):
    """
    Set pixels to random color where an annotation exists
    This is to show what is already annotated
    """
    new_image = deepcopy(image)
    for ann in annotations: 
        new_image[ann['segmentation']]=[random.randint(a,b),random.randint(a,b),random.randint(a,b)]
    return new_image


def initAnnotation(mask, cat=1):
    '''
    Initialize an annotation dictionary.
    WARNING:"id" and "image_id" still missing!
    '''
    annotation = {
        'segmentation': mask.astype(bool),
        'area': updateArea(mask),
        'iscrowd': 0,
        'bbox': updateBbox(mask),
        'category_id': cat,
    }
    return annotation


def captureMask(image, predictor, list_xy, list_label, p=5):
    '''
    Generate a mask using predictor and a collection of click prompt inputs.
    An image displaying it together with its corresponding bbox is saved to {path_file_SA_tmp_img}
    The mask is returned as output
    '''
    input_points = np.array(list_xy)
    input_labels = np.array(list_label)
    
    mask,_,_ = predictor.predict(
        point_coords=input_points,
        point_labels=input_labels,
        multimask_output=False
        )
    
    mask = np.asanyarray(mask.squeeze(), dtype=np.uint8)
    [[x_start,x_stop],[y_start,y_stop]] = bboxLocation(initAnnotation(mask))
    rect = cv2.rectangle(image.copy(), (x_start,y_start), (x_stop,y_stop), (255,0,0), 5)
    
    plt.figure()
    plt.imshow(image)
    plt.imshow(rect)
    plt.axis('off')
    show_mask(mask, plt.gca())
    show_points(input_points, input_labels, plt.gca())
    plt.savefig(path_file_SA_tmp_img, bbox_inches='tight')
    plt.close()
    
    return mask


def mouse_callback(event, x, y, flags, param):
    '''
    Record click prompt from user
    '''
    global _clk_, _xy_clk_, _label_clk_
    
    if event == cv2.EVENT_LBUTTONDOWN:
        _label_clk_.append(1)
        _xy_clk_.append([int(float(x)*_scaling_factor_x_), int(float(y)*_scaling_factor_x_)])
        _clk_ = True
    
    elif event == cv2.EVENT_RBUTTONDOWN:
        _label_clk_.append(1 if _label_clk_==[] else 0)
        _xy_clk_.append([int(float(x)*_scaling_factor_x_), int(float(y)*_scaling_factor_x_)])
        _clk_ = True


def userInput(displayed_image, ref_image, display_width=display_width):
    '''
    Display image saved at {path_file_SA_tmp_img} and wait for any input: click or action
    '''
    global _clk_,_scaling_factor_x_
    
    # reset click token
    _clk_ = False
    
    # resize image to display it
    hw_image = ref_image.shape[:2]
    height = hw_image[0]
    width = hw_image[1]
    _scaling_factor_x_ = width/display_width
    resized_image = cv2.resize(displayed_image,(display_width,int(height/_scaling_factor_x_)))
    
    # display image
    window_name = "Image"
    cv2.namedWindow(f'{window_name}')
    cv2.setMouseCallback(f'{window_name}', mouse_callback)
    cv2.imshow(f'{window_name}', resized_image[:,:,::-1])
    
    # exit condition
    while True:
        key = cv2.waitKey(0)
        key = chr(key)
        # return action to perform if user did not click 
        action = key if not _clk_ and key in ['w','q','s','f','c','d','a','l'] else 0
        break
    
    # terminate
    cv2.destroyAllWindows()
    return action


def getSingleAnnotationSA(image,predictor,cat=1):
    '''
    Returns one annotation for one image
    i) If the displayed image shows clicks together with the corresponding mask:
        - press 'w' to remove last click
        - press 'q' to remove all clicks
        - press 's' to validate current mask and add it to temporary list existing at the upper level: getAnnotationsSA.
        - press 'f' to pop the corresponding mask from json file
        
    ii) The displayed image shows no click and some validated temporary masks (or not):
        - press 'c' to remove last validated mask from list existing at the upper level: getAnnotationsSA.
        - press 'd' to save them to json file and move to next image.
        - press 'a' to save them to json file and move to previous image.
        - press 'l' to save them to json file and quit annotation process.
    '''
    global _xy_clk_ ,_label_clk_
    
    _xy_clk_ = []
    _label_clk_ = []
    ann = None
    
    while True:
        # no clicks recorded
        if len(_xy_clk_)==0:
            # collect initial clicks or action
            action = userInput(image.copy(),image)
            
            # exit condition when no points recorded
            if action in ['c','a','d','l']: break
        
        if len(_xy_clk_)>0:
            # compute mask corresponding to recorded clicks
            mask = captureMask(image,predictor,_xy_clk_,_label_clk_)
            # display and wait for user action
            action = userInput(imreadRGB(path_file_SA_tmp_img),image)
            
            # 'w' for remove: clear last click
            if action == 'w':
                if len(_xy_clk_)==1: action = 'q'
                else:
                    _xy_clk_ = _xy_clk_[:-1]
                    _label_clk_ = _label_clk_[:-1]
                
            # 'q' for reset all clicks: empty recorded clicks
            if action == 'q':
                _xy_clk_ = []
                _label_clk_ = []
                
            # 's' for validation: create an annotation
            # 'f' for pop: pop the annotation from json file
            elif action in ['s','f']:
                ann = initAnnotation(mask,cat=cat)
                break
                
    # action value and single annotation is passed to getAnnotationsSA
    return action, ann


def getAnnotationsSA(image,predictor,offset,cat=1):
    '''
    Get all annotations for one image
    '''
    
    # container for new single annotations 
    image_annotations = []
    
    while True:
        action, single_annotation = getSingleAnnotationSA(hideFromImage(image_annotations,image),predictor,cat=cat)
        
        # a single annotation has been validated
        if action == 's':
            # add annotation id
            single_annotation['id'] = len(image_annotations)+offset
            # add annotation to temporarily list
            image_annotations.append(single_annotation)
            
        # cancellation request from the user 
        elif action == 'c':
            if len(image_annotations)>0:
                # remove last annotation from temporarily list
                image_annotations = image_annotations[:-1]
                
        # need to go to upper level to either exit annotation process or modify json file
        elif action in ['f','l','a','d']: break    
        
    return action, image_annotations, single_annotation


def annotateSA(path_dir_anns,path_list_images,cat,predictor=SamPredictor(initSam())):
    '''
    Get annotations for all images existing in json_file
    '''
    to_be_updated = []
    #init
    k=0
    while True:
        # get id, read and set SA predicator
        path_file_image = path_list_images[k]
        timestamp_id = getId(path_file_image)  
        image = imreadRGB(path_file_image)
        predictor.set_image(image)
        
        path_file_anns = join(path_dir_anns,timestamp_id+'.json')
        existing_anns = decodeMasks(readJson(path_file_anns)) if exists(path_file_anns) else []
        
        # run annotation process
        action, new_annotations, single_annotation = getAnnotationsSA(hideFromImage(existing_anns,image),predictor, offset=len(existing_anns),cat=cat)
        
        # save temporarily annotations
        if action in ['a','d','f','l']:
            if len(new_annotations)>0:
                existing_anns+=new_annotations
                writeJson(path_file_anns,encodeMasks(existing_anns))
                to_be_updated.append(path_file_anns)
        
        # move to next image
        if action == 'd': k = k+1 if k+1<len(path_list_images) else 0
        # back to previous image
        elif action == 'a': k = k-1 if -(k-1)<len(path_list_images) else 0
        # find matching annotation to pop
        elif action == 'f' and exists(path_file_anns):
            # reload and update
            previous_anns = existing_anns
            existing_anns = [ann for ann in previous_anns if not IoU(ann,single_annotation)>0.9]
            if len(previous_anns)>len(existing_anns):
                for ann_id,ann in enumerate(existing_anns): ann['id'] = ann_id
                writeJson(path_file_anns,encodeMasks(existing_anns))
                to_be_updated.append(path_file_anns)
            
        # move to next image or quit
        elif action == 'l':
            if exists(path_file_SA_tmp_img): remove(path_file_SA_tmp_img)
            break
        
    return list(set(to_be_updated))