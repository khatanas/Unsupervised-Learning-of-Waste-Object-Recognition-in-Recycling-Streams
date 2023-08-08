import matplotlib.pyplot as plt
import cv2
import numpy as np
from random import randrange
from win32api import GetSystemMetrics

# global variables used to store the clicks, and the triplets of clicks
clicks = list()
triplets = list()

""" This function will be called whenever the mouse is clicked """
def mouse_callback(event, x, y, flags, params):
    global clicks
    global triplets

    # left-click event value is 1
    if event == 1:
        # store the coordinates of the right-click event
        clicks.append([x, y])
        
        # keep track of the clicks
        if len(clicks) == 1: print('p2', end=' ')
        if len(clicks) == 2: print('mask', end=' ')
        
        # click 1,2 define the hyperplane, click 3 indicates the halfplane to mask
        # the triplet is stored in triplets and clicks list is reinitialized
        if len(clicks) == 3: 
            triplets.append(clicks)
            clicks = list()
            print('p1', end=' ')

""" This function open a new window and display the img located at img_path. 
    By clicking on the image, regions to mask are defined.
    Click n°1,2 define the hyperplane, click n°3 indicates the halfplane to mask.
    Once you are done, press 0."""
def get_triplets(path):
    # define new window
    img = cv2.imread(path,0)
    scale_width = 640 / img.shape[1]
    scale_height = 480 / img.shape[0]
    scale = min(scale_width, scale_height)
    window_width = int(img.shape[1] * scale)
    window_height = int(img.shape[0] * scale)
    cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('image', window_width, window_height)
    print('p1', end=' ')
    
    #set mouse callback function for window
    cv2.setMouseCallback('image', mouse_callback)

    # show and wait for 0 key
    cv2.imshow('image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
""" This function will generate the final mask """
def generate_mask(path, triplets=triplets):
    # call the function that defines triplets
    get_triplets(path)

    # load img
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    
    # get dimensions
    dim = np.array(img.shape[0:2])
    mask = np.ones(dim)

    # for each triplet
    for triplet in triplets:
        p1 = np.array(triplet.pop(0))
        p2 = np.array(triplet.pop(0))
        bk = np.array(triplet.pop(0))
        
        # compute a,b parameters (y=a*x+b)
        if (p2-p1)[0] == 0: a = (p2-p1)[1]
        else: a = (p2-p1)[1]/(p2-p1)[0]
        b = p1[1]-a*p1[0]
        
        # generate single mask
        pxl = np.indices(dim)
        tmp = a*pxl[1]+b
        tmp = (tmp>pxl[0]).astype(int)

        # invert 0 and 1 if masking is inverted
        if tmp[bk[1],bk[0]] == 1:
            tmp = -(tmp-1)
            
        # merge for final mask
        mask *= tmp

    # reset
    triplets.clear()
    return mask.astype(np.uint8)

""" This function apply the final mask to the img and 
    crop it to keep only the interesting part """
def refine_img(path,mask):
    # load img
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # get crop positions
    locx=np.indices(mask.shape)[1]
    xmin = min(locx[mask>0])
    xmax = max(locx[mask>0])
    # apply final mask and crop img
    img *= mask[:,:,np.newaxis]
    img = img[:,xmin:xmax,:]
    cropped_mask = mask[:,xmin:xmax]
    return img,cropped_mask

def square_img(img, p):
    # make it square
    h = img.shape[0]
    w = img.shape[1]
    c = img.shape[2]

    if h<w:
        tmp = np.zeros([w,w,c],dtype=np.uint8)
        rdm_start = randrange(0,w-h)
        tmp[rdm_start:rdm_start+h,:,:] = img
        img = tmp

    elif w<h:
        rdm_start = randrange(0,h-w)
        img = img[rdm_start:rdm_start+w,:,:]

    img = cv2.resize(img,(p,p))
    
    return img