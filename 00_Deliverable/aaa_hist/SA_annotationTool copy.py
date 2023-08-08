from helper._lib_SA import *
from helper._annotations import encodeMasks,decodeMasks,updateArea,updateBbox,bboxLocation
from helper._tuning import *


from copy import deepcopy
from os import remove
from os.path import exists



path_file_SA_tmp_img = join(path_lib_SA,'tmp_mask.png')
display_width = 1000
#################### specific to annotation tool ##################################
def hideFromImage(annotations,image,value=120):
    new_image = deepcopy(image)
    for ann in annotations:
        new_image[ann['segmentation']]=value
    return new_image


def initAnnotation(mask, cat=1):
    '''
    Creates an encoded annotation from binary mask (bool or 0/1)
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
    Same as getMask but slightly modified for the annotation tool
    Once the maks is computed, the corresponding bbox is created
    A plot displaying the image, the bbox, the mask, the prompts is created but not shown.
    Instead, the plot is saved to a tmp location for future use
    '''
    input_point = np.array(list_xy)
    input_label = np.array(list_label)
    
    mask,_,_ = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
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
    show_points(input_point, input_label, plt.gca())
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
        if _label_clk_==[]: _label_clk_ = [1]
        else:_label_clk_.append(0)
        _xy_clk_.append([int(float(x)*_scaling_factor_x_), int(float(y)*_scaling_factor_x_)])
        _clk_ = True


def newPrompt(displayed_image, ref_image, display_width=display_width):
    '''
    Display image and wait for any input: clicks or keyboard input
    '''
    global _clk_,_scaling_factor_x_
    
    hw_image = ref_image.shape[:2]
    height = hw_image[0]
    width = hw_image[1]
    
    _scaling_factor_x_ = width/display_width
    resized_image = cv2.resize(displayed_image,(display_width,int(height/_scaling_factor_x_)))
    
    window_name = "Define new input prompts, then press any key"
    cv2.namedWindow(f'{window_name}')
    cv2.setMouseCallback(f'{window_name}', mouse_callback)
    cv2.imshow(f'{window_name}', resized_image[:,:,::-1])
    
    while True:
        key = cv2.waitKey(0)
        key = chr(key)
        # remove, reset, validate, cancel, done, quit
        output = key if key in ['r','x','v','c','d','q'] else 0
        break

    cv2.destroyAllWindows()
    return output


def getSingleAnnotationSA(image,predictor,cat=1):
    '''
    Returns one annotation for one image
    When the image is displayed you can
    - click left to add (+) point
    - click right to add (-) point (unless first click, which is always (+))
    - press 'v' to validate current annotation
    - press 'r' to remove last click
    - press 'x' to remove all displayed clicks
    - press 'c' to cancel previous annotation
    - press 'q' to quit annotation process
    - press 'd' to validate current image (done) and display next one
    An annotation is added to the json file after each validation. 
    Pressing 'r' can only remove the annotation of the current annotation session 
    '''
    global _clk_, _xy_clk_ ,_label_clk_
    
    _xy_clk_ = []
    _label_clk_ = []
    ann = None
    
    image_tmp = image.copy()

    # annotation level
    while True:
        # re/set global variable _clk_
        _clk_ = False
        # display image and wait for input
        output = newPrompt(image_tmp,image)
        
        # clicked to update annotation
        if _clk_:
            # compute new mask (and save temporary result to display it)
            mask = captureMask(image,predictor,_xy_clk_,_label_clk_)
            # load to display current mask
            image_tmp = imreadRGB(path_file_SA_tmp_img)
        
        # remove last click
        elif output == 'r':
            # need at least a click 
            if len(_xy_clk_)<=1:
                _xy_clk_ = []
                _label_clk_ = []
                image_tmp = image.copy()
                continue
            else:
                _xy_clk_ = _xy_clk_[:-1]
                _label_clk_ = _label_clk_[:-1]
                # compute new mask (and save temporary result to display it)
                mask = captureMask(image,predictor, _xy_clk_,_label_clk_)
                # load to display current mask
                image_tmp = imreadRGB(path_file_SA_tmp_img)
            
        # reset all clicks
        elif output == 'x':
            _xy_clk_ = []
            _label_clk_ = []
            image_tmp = image.copy()
        
        # validation possible only if clicked at least once
        elif output == 'v':
            if len(_xy_clk_)>0: 
                ann = initAnnotation(mask,cat=cat)
                break
            else: output = 0
            
        # user pressed { 'c', 'q' ,'d'}
        elif output !=0: break
    
    # output value and single annotation is passed to getAnnotationsSA
    return output, ann



def getAnnotationsSA(image,predictor,offset,cat=1):
    '''
    Get all annotations for one image
    getSingleAnnotation is called until user:
    - press 'q' to quit annotation process
    - press 'd' to validate current image (done) and display next one
    '''

    # container for new single annotations 
    image_annotations = []
    
    while True:
        output, single_annotation = getSingleAnnotationSA(hideFromImage(image_annotations,image),predictor,cat=cat)
        
        # a single annotation has been validated
        if output == 'v':
            # add annotation id
            single_annotation['id'] = len(image_annotations)+offset
            image_annotations.append(single_annotation)
            
        # cancellation request from the user 
        elif output == 'c':
            if len(image_annotations)>0:
                image_annotations = image_annotations[:-1]
            
        # user pressed {'q' ,'d'}
        elif output!=0: break    
        
    # output value and annotation collection is passed to annotateSA
    return output, image_annotations


def annotateSA(path_file_json,path_dir_images,predictor,cat=1):
    '''
    Get annotations for all image in path_file_json
    The annotations already existing are displayed and the new one are added to the file
    You can quit the annotation process:
    - of one image by pressing 'd'
    - the main one by pressing 'q'
    '''
    # load json file
    
    json_file = readJson(path_file_json)
    existing_annotations = decodeMasks(json_file['annotations'])
    
    # annotate each image
    for image_field in json_file['images']:
        # read and set SA predicator
        image = imreadRGB(join(path_dir_images,image_field['file_name']))
        predictor.set_image(image)
        
        image_id = image_field['id']        
        image_anns = [ann for ann in existing_annotations if ann['image_id']==image_id]
        
        # run annotation process
        output, new_annotations = getAnnotationsSA(hideFromImage(image_anns,image),predictor, offset=len(existing_annotations),cat=cat)
        
        # add new annotations
        for ann in new_annotations:
            ann['image_id'] = image_id
        
        # save to new file
        existing_annotations+=new_annotations
        json_file['annotations'] = encodeMasks(existing_annotations)
        writeJson(path_file_json,json_file)
        
        # abort annotation process and quit
        if output == 'q': 
            if exists(path_file_SA_tmp_img): remove(path_file_SA_tmp_img)
            break
    
    return json_file