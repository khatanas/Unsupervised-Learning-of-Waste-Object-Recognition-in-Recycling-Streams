import os
from os.path import join, isfile
from os import listdir
import shutil
import json
import random
from random import shuffle
'''import pandas as pd
import numpy as np'''
import cv2

def writeJson(path2file,jsn):
    with open(path2file, 'w') as f:
        json.dump(jsn,f)

def readJson(path2file):
    with open(path2file, 'r') as f:
        jsn = json.load(f)
    return jsn

def initDict(path2imgs,offset=0,path2file=False):
    '''
    Creates a json file for the images located at path2imgs and save it at path2file
    The offset allows to add an offset to the "images"[id] field
    '''
    empty_dict = {
        'info':{},
        'licenses':[],
        'images':[],
        'categories':[
            {
                'supercategory':'fg',
                'id':1,
                'name':'fg'
            }
        ],
        'annotations':[]
    }
    
    for idx,d in enumerate(sorted(listdir(path2imgs))):
        image = cv2.imread(join(path2imgs,d))
        height = image.shape[0]
        width = image.shape[1]
        tmp = {
            'file_name': d,
            'height': height,
            'width':width,
            'id':idx+offset
        }
        empty_dict['images'].append(tmp)
        
    if path2file != False:
        with open(path2file, 'w') as f:
            json.dump(empty_dict, f)
        print(f'json created at {path2file}')
        
    return empty_dict

def clearAnnotations(path2file):
    '''
    Empty the "annotation" field of json file located at path2file
    '''
    with open(path2file, 'r') as f:
        tmp_json = json.load(f)
    tmp_json['annotations'] = []
    with open(path2file, 'w') as f:
        json.dump(tmp_json, f)

def trainTestSplit(path2file,nb,test_list=False):
    '''
    Perform a train-test split with "nb" images in the test file.
    If a test_list is provided, containing the name of some images, the split is performed 
    based on the provided list and the size of the test part becomes len(test_list)
    Creates te_{json_name}.json and tr_{json_name}.json located at path2file
    '''
    with open(path2file, 'r') as f:
        json_file=json.load(f)
    
    if not test_list:
        test_list = sorted([d['file_name'] for d in json_file['images']])
        random.seed(10)
        shuffle(test_list)
        test_list = sorted(test_list[:nb])
    
    imgs_tr = [d for d in json_file['images'] if d['file_name'] not in test_list]
    imgs_te = [d for d in json_file['images'] if d['file_name'] in test_list]

    id_tr = [d['id'] for d in imgs_tr]
    id_te = [d['id'] for d in imgs_te]

    anns_tr = [d for d in json_file['annotations'] if d['image_id'] in id_tr]
    anns_te = [d for d in json_file['annotations'] if d['image_id'] in id_te]

    json_tr = {
        'info': json_file['info'],
        'licenses': json_file['licenses'],
        'images': imgs_tr,
        'categories': json_file['categories'],
        'annotations': anns_tr
        }

    json_te = {
        'info': json_file['info'],
        'licenses': json_file['licenses'],
        'images': imgs_te,
        'categories': json_file['categories'],
        'annotations': anns_te
        }

    json_name = path2file.split('/')[-1]
    loc = path2file.replace(f'/{json_name}','')
    with open(join(loc,f'tr_{json_name}'), 'w') as f:
        json.dump(json_tr, f)

    with open(join(loc,f'te_{json_name}'), 'w') as f:
        json.dump(json_te, f)
        
def bboxArea(ann):
    return ann['bbox'][2]*ann['bbox'][3]
