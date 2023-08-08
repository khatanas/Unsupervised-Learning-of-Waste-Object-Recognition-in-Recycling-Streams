import os
from os.path import join, isfile
from os import listdir
import shutil
import json

def renameImgs(path2imgs,offset=0):
    '''
    Rename images: {00000.extension, 00001.extension, ...}
    An offset can be added
    '''
    names = sorted(listdir(path2imgs))
    for idx,name in enumerate(names):
        extension = name.split('.')[-1]
        new_name = f"{str(idx+offset).zfill(5)}.{extension}"
        
        previous_path = join(path2imgs,name)
        new_path = join(path2imgs,new_name)
        
        shutil.move(previous_path,new_path)
        
        
        
def toTrainSubfolders(src,destination,img_per_folder,sublist=False,
                        header='',letter='A',nb=0,nb_max=2):
    '''
    Maskcut helper: copy imgs located at src to specified destination. If a sublist is provided, send only the sublist. 
    The imgs are stored in a set of folder containing img_per_folder imgs each.
    '''
    
    # get list of name
    if sublist==False: imgs = sorted([f for f in listdir(src) if (isfile(join(src,f)) and (f.endswith('png') or f.endswith('jpg')))])
    else: imgs = sorted([f for f in listdir(src) if f in sublist])
    # form list of sublists of name (len(sublist)=img_per_folder)
    repartition = []
    while len(imgs)>img_per_folder:
        repartition.append(imgs[:img_per_folder])
        del imgs[:img_per_folder]
    repartition.append(imgs)
    del imgs
    # copy each group into subfolder
    for group in repartition:
        # generate subfolder
        if nb >= int('9'*nb_max):
            letter = chr(ord(letter)+1)
            nb = 0
        folder = letter+str(nb).zfill(nb_max)
        if header != '': folder = header+'_'+folder
        newpath=join(destination,folder)
        os.makedirs(newpath)
        nb += 1
        
        # copy file
        for elem in group:
            shutil.copy(join(src,elem),join(newpath,elem))
            
            
            
def adaptMaskCutGT(path2file,path2imgs):
    '''
    Images sent to MaskCut are divided into subfolder, which appear in the "file name" field
    adaptMaskCutGT takes as input the location of the json file from MaskCut and the location of the images
    It removes the subfolders from the name, and sorts the images 
    '''
    
    # load json as string, remove subfolder name, and overwrite the json file
    with open(path2file, 'r') as file:
        dict_mc = file.read()
        
    folder_names = listdir(path2imgs)
    for f in folder_names:
        dict_mc = dict_mc.replace(f'{f}/','')
    dest = join(path2file)
    with open(dest, 'w') as file:
        file.write(dict_mc)
        
    # open as json, sort the images by "file name", update the annotation id, and overwrite json
    with open(path2file, 'r') as f:
        dict_mc= json.load(f)    
    dict_mc['images'] = sorted(dict_mc['images'], key = lambda x: x['file_name'])
    
    # update image id in "image" as well as in "annotations"
    dict_id = {}    
    for idx,image_field in enumerate(dict_mc['images']):
        # memorize previous id
        id = image_field['id']
        dict_id[id]=idx
        #replace it
        image_field['id'] = idx
    # adapt annotation
    for ann in dict_mc['annotations']:
        ann['image_id'] = dict_id[ann['image_id']]
    
    with open(path2file, 'w') as f:
        json.dump(dict_mc, f)
        
    print('Search and replace operation completed successfully!')
    
    
    
def adaptBlenderGT(path2file):
    '''
    Correct the "categories" field returned by Blender process
    Takes as input the location of the json from Blender, corrects it, and overwrite it
    '''
    
    # open file and update "categories" field
    with open(path2file, 'r') as f:
        dict_blender = json.load(f)
        
    dict_blender['categories'] =[
        {
            "supercategory": "fg",
            "id": 1,
            "name": "fg"
        }
    ]

    # change annotation category accordingly
    for ann in dict_blender['annotations']:
        ann['category_id'] = 1
    
    # Open the output file in write mode
    with open(path2file, 'w') as f:
        json.dump(dict_blender, f)

    print('Search and replace operation completed successfully!')