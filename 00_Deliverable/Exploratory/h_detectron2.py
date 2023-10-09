p2d2 = '/home/dkhatanassia/detectron2'
p2mc = '/home/dkhatanassia/CutLER/maskcut'
p2cl = '/home/dkhatanassia/CutLER/cutler'

import sys
sys.path.append(p2d2)
sys.path.append(p2mc)
sys.path.append(p2cl)

import os
os.chdir(p2mc)
from colormap import random_color, random_colors

'''import detectron2
from detectron2 import model_zoo
from detectron2.utils.logger import setup_logger
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg'''
from detectron2.utils.visualizer import Visualizer
from detectron2.data.catalog import MetadataCatalog, DatasetCatalog
from detectron2.data.datasets import register_coco_instances

import matplotlib.pyplot as plt
import cv2
from os.path import join


def registerCatalog(name,path2json,path2imgs):
    '''
    Creates a detectron2 catalog called {name} gathering info
    from json located at path2json and images located at path2imgs
    '''
    if name in DatasetCatalog.list():
        DatasetCatalog.remove(name)
        MetadataCatalog.remove(name)
    register_coco_instances(name, {}, path2json, path2imgs)
    print(f'registered: {name}\njson is: {path2json}\nimgs are at: {path2imgs}\n')
    
    
def visualizeSample(sample, name):
    '''
    Outputs the {sample}-th image of the catalog {name} together with its annotations 
    '''
    catalog = DatasetCatalog.get(name)
    meta = MetadataCatalog.get(name)
    d = catalog[sample]
    d_name = d['file_name'].split('/')[-1]

    # use detectron2 Visualizer
    img = cv2.imread(d["file_name"])
    count = len(d["annotations"])
    visualizer = Visualizer(img[:, :, ::-1], metadata=meta, scale=0.5)
    out = visualizer.draw_dataset_dict(d)
    plt.title(f'{d_name}: {count} annotations')
    plt.imshow(out.get_image())
    plt.show()


def visualizePseudoGT(sample, json_name_previous, json_name_new, p=15, saveto=False):
    '''
    Creates a 1x3 plot diplaying:
    i. the image without annotations
    ii. the image with annotation of catalog named {json_name_previous}
    iii. the image with annotation of catalog named {json_name_new}
    if "saveto" parameter is an existing path, the plot is saved at the given location
    '''
    jsons = [json_name_previous, json_name_new]

    # load catalogs
    catalogs = [DatasetCatalog.get(jsons[i]) for i in range(len(jsons))]
    metas = [MetadataCatalog.get(jsons[i]) for i in range(len(jsons))]
                
    name = catalogs[0][sample]['file_name'].split('/')[-1]
    # read imgs
    img = cv2.imread(catalogs[0][sample]['file_name'])
    img=img[:,:,::-1]
    visualizer = [Visualizer(img, metadata=metas[i], scale=0.5) for i in range(len(jsons))]
    out = [visualizer[i].draw_dataset_dict(catalogs[i][sample]).get_image() for i in range(len(jsons))]
    
    # Define the size of the figure
    fig, axs = plt.subplots(1, 3, figsize=(p, 3*p))
    axs = axs.flatten()
    for i, ax in enumerate(axs):
        if i==0:
            ax.imshow(img)
            ax.set_title(f'input: {name}')
        elif i <= len(out):
            ax.imshow(out[i-1])
            ax.set_title(jsons[i-1])
        # If there are no more images, turn off the axis to leave it blank
        else:
            ax.axis("off")

    if saveto!=False: plt.savefig(join(saveto,f'visu_{name}'), bbox_inches='tight')
    
    plt.tight_layout()
    plt.show()