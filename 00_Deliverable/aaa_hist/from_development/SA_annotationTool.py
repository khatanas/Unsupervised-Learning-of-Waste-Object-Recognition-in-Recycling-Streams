from h_SA import *

#################### specific to annotation tool ##################################
def annotationExNihilo(binary_mask):
    '''
    Creates an encoded annotation from binary mask (bool or 0/1)
    WARNING:"id" and "image_id" still missing!
    '''
    rle = maskUtils.encode(np.asfortranarray(binary_mask))
    area = int(maskUtils.area(rle))
    
    rle['counts'] = rle['counts'].decode('utf-8')
    annotation = {
        'segmentation': rle,
        'area': area,
        'iscrowd': 0,
        'bbox': box_xyxy_to_xywh(batched_mask_to_box(torch.from_numpy(binary_mask.astype(bool)))).tolist(),
        'category_id':1,
    }
    
    return annotation


def getMask2(image,predictor, xy_list,label_list,p=5):
    '''
    Same as getMask but slightly modified for the annotation tool
    Once the maks is computed, the corresponding bbox is created
    A plot displaying the image, the bbox, the mask, the prompts is created but not shown.
    Instead, the plot is saved to a tmp location for future use
    '''
    input_point = np.array(xy_list)
    input_label = np.array(label_list)

    mask, score, logit = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        multimask_output=False
        )
    
    mask = mask.squeeze()
    bbox = box_xyxy_to_xywh(batched_mask_to_box(torch.from_numpy(mask))).tolist()
    rect = cv2.rectangle(image.copy(), (bbox[0],bbox[1]), (bbox[0]+bbox[2],bbox[1]+bbox[3]), (0,0,0), 5)
    #print((bbox[0],bbox[1]), (bbox[0]+bbox[2],bbox[1]+bbox[3]))

    plt.figure(figsize=(8,6))
    plt.imshow(image[:,:,::-1])
    plt.imshow(rect)
    plt.axis('off')
    show_mask(mask, plt.gca())
    show_points(input_point, input_label, plt.gca())
    plt.savefig('tmp_mask.png',bbox_inches='tight')
    plt.close()
    
    binary_mask = np.asarray(mask, dtype=np.uint8)

    return binary_mask


def mouse_callback(event, x, y, flags, param):
    '''
    Record prompt from user
    '''
    global clk, xy_clk, label_clk
    
    if event == cv2.EVENT_LBUTTONDOWN:
        label_clk.append(1)
        xy_clk.append([int(x*scaling_factor_x), int(y*scaling_factor_x)])
        clk = True
    
    elif event == cv2.EVENT_RBUTTONDOWN:
        if label_clk==[]: label_clk = [1]
        else:label_clk.append(0)
        xy_clk.append([int(x*scaling_factor_x), int(y*scaling_factor_x)])
        clk = True


def inputSA(displayed_image,image):
    '''
    Display image and wait for any input: clicks or keyboard input
    '''
    global clk,scaling_factor_x
    
    display_x = 1000
    scaling_factor_x = image.shape[1]/display_x
    resized_image = cv2.resize(displayed_image,(display_x,int(image.shape[0]/scaling_factor_x)))
    
    cv2.namedWindow("Image")
    cv2.setMouseCallback("Image", mouse_callback)
    cv2.imshow("Image", resized_image)
    
    while True:
        key = cv2.waitKey(0)
        key = chr(key)
        # remove, reset, validate, cancel, done, quit
        output = key if key in ['r','x','v','c','d','q'] else 0
        break

    cv2.destroyAllWindows()
    return output


def getSingleAnnotationSA(image,predictor):
    '''
    Returns one annotation for one image
    When the image is displayed you can
    - click left to add (+) point
    - click right to add (-) point (unless first click, which is always (+))
    - press 'v' to validate current annotation
    - press 'c' to cancel previous annotation
    - press 'r' to remove last click
    - press 'x' to remove all displayed clicks
    - press 'q' to quit annotation process
    - press 'd' to validate current image (done) and display next one
    An annotation is added to the json file after each validation. 
    Pressing 'r' can only remove the annotation of the current annotation session 
    '''
    global clk, xy_clk ,label_clk
    
    xy_clk = []
    label_clk = []
    ann = None
    
    image_tmp = image.copy()

    # annotation level
    while True:
        # re/set global variable clk
        clk = False
        # display image and wait for input
        output = inputSA(image_tmp,image)
        
        # clicked to update annotation
        if clk:
            # compute new mask (and save temporary result to display it)
            mask = getMask2(image,predictor,xy_clk,label_clk)
            # load to display current mask
            image_tmp = cv2.imread('/home/dkhatanassia/segment-anything/tmp_mask.png')
        
        # remove last click
        elif output == 'r':
            # need at least a click 
            if len(xy_clk)<=1:
                xy_clk = []
                label_clk = []
                image_tmp = image.copy()
                continue
            else:
                xy_clk = xy_clk[:-1]
                label_clk = label_clk[:-1]
                # compute new mask (and save temporary result to display it)
                mask = getMask2(image,predictor, xy_clk,label_clk)
                # load to display current mask
                image_tmp = cv2.imread('/home/dkhatanassia/segment-anything/tmp_mask.png')
            
        # reset all clicks
        elif output == 'x':
            xy_clk = []
            label_clk = []
            image_tmp = image.copy()
        
        # validation possible only if clk at least once
        elif output == 'v':
            if len(xy_clk)>0: 
                ann = annotationExNihilo(mask)
                break
            else: output = 0
            
        # user pressed { 'c', 'q' ,'d'}
        elif output !=0: break
            
    return output, ann


def getAnnotationsSA(image,predictor,offset):
    '''
    Get all annotations for one image
    getSingleAnnotation is called until user:
    - press 'q' to quit annotation process
    - press 'd' to validate current image (done) and display next one
    '''
    image_tmp = [image.copy()]
    k = 0
    anns = []
    while True:
        output, ann = getSingleAnnotationSA(image_tmp[k],predictor)
        if output == 'v':
            ann['id'] = len(anns)+offset
            anns.append(ann)
            
            tmp_mask = (maskUtils.decode(ann['segmentation'])).astype(bool)
            #tmp_mask = ann['segmentation']
            tmp_img = image_tmp[k].copy()
            tmp_img[tmp_mask] = 120
            image_tmp.append(tmp_img)
            k+=1
                        
        elif output == 'c':
            if len(anns)>0:
                anns = anns[:-1]
                image_tmp = image_tmp[:-1]
                k-=1
            
        # user pressed {'q' ,'d'}
        elif output!=0: break    
    return output, anns


def annotateSA(path2file,path2imgs,predictor):
    '''
    Get annotations for all image in path2file
    The annotations already existing are displayed and the new one are added to the file
    You can quit the annotation process:
    - of one image by pressing 'd'
    - the main one by pressing 'q'
    '''
    # load json file
    with open(path2file, 'r') as f:
        jsn= json.load(f)
        
    # annotate each image
    for im in jsn['images']:
        # read and set SA predicator
        image = cv2.imread(join(path2imgs,im['file_name']))
        predictor.set_image(image)
        
        # remove already existing masks from image to avoid duplicates
        tmp_image = image.copy()
        image_id = im['id']
        tmp_ann = [d['segmentation'] for d in jsn['annotations'] if d['image_id']==image_id]
        for t in tmp_ann:
            # retrieve rle from str and convert rle to mask
            t['counts'] = t['counts'].encode('utf-8')
            tmp_mask = maskUtils.decode(t).astype(bool)
            tmp_image[tmp_mask]=120
            t['counts'] = t['counts'].decode('utf-8')
            
        # run annotation process
        output, anns = getAnnotationsSA(tmp_image,predictor,len(jsn['annotations']))
        
        # add new annotations
        for ann in anns:
            ann['image_id'] = image_id
        jsn['annotations'] += anns
        
        # save new file
        with open(path2file, 'w') as f:
            json.dump(jsn, f)
        
        # abort annotation process and quit
        if output == 'q': break
    
    return jsn