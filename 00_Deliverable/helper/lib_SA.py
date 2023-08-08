from config.paths import *
from config.parameters import arch_SA

from helper.paths import getImagePath
from helper.tuning import imreadRGB
from helper.common_libraries import np,join,plt,random

from segment_anything import sam_model_registry
############### Directly from {SA,detectron2} github or slighty modified ########################

def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:,:,3] = 0
    random.seed(10)
    for ann in sorted_anns:
        m = ann['segmentation']
        random.seed(10)
        color_mask = np.concatenate([np.random.random(3), [0.5]])
        img[m] = color_mask
    ax.imshow(img)

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    
def show_points(coords, labels, ax, marker_size=20):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='o', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='o', s=marker_size, edgecolor='white', linewidth=1.25)   

def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))    


############### perso ########################
def initSam():
    model_type = arch_SA
    device = "cuda"
    sam = sam_model_registry[model_type](checkpoint=path_file_SA_checkpoint)
    sam.to(device=device)
    return sam
    
    
def getMask(image, predictor, xy_list,label_list,show=False,p=5):
    '''
    Takes as input the image, the predictor, the prompts
    Returns the resulting mask together with info
    '''
    input_point = np.array(xy_list)
    input_label = np.array(label_list)

    mask, score, logit = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        multimask_output=False
        )
    
    if show:
        plt.figure(figsize=(p,p))
        plt.imshow(image)
        show_mask(mask, plt.gca())
        show_points(input_point, input_label, plt.gca())
        plt.title(f"best mask, score: {float(score):.2f}", fontsize=14)
        plt.axis('on')
        plt.show()  
        
    return mask.squeeze(), score, logit


def visualizeSA(image, masks=[],
                title='',show_title=True ,axis='off',path_save=False):
    '''
    Display colored masks on image
    '''
    if type(masks) is not list: masks = [masks]
    
    plt.figure()
    plt.imshow(image)
    if show_title: plt.title(title)
    show_anns(masks)
    plt.axis(axis)
    if path_save!=False: plt.savefig(join(path_save,f'{title}.jpg'),bbox_inches='tight')
    plt.show()



def visualizeSA2(masks=[],
                title='',show_title=True ,axis='off',path_save=False):
    '''
    Display colored masks on image
    '''
    if type(masks) is not list: masks = [masks]
    image = imreadRGB(getImagePath(masks[0]['timestamp_id']))
    
    plt.figure()
    plt.imshow(image)
    if show_title: plt.title(title)
    show_anns(masks)
    plt.axis(axis)
    if path_save!=False: plt.savefig(join(path_save,f'{title}.jpg'),bbox_inches='tight')
    plt.show()