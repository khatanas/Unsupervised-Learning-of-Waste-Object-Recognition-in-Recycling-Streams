from helper._config_paths import *
from helper._config_classification import *
from helper._scriptIO import *
from helper._CLIP import *
from helper._tuning import readJson,writeJson, imreadRGB
from helper._annotations import decodeMasks,cropMask,getHeads
from helper._paths import getId, getImagePath,initPath

from os import listdir
############################################## CONFIG #######################################################################

# get list of images, masks
points_per_side, path_dir_src = requestPtsPerSide(path_root_masks_from_images)
channel, path_dir_src = requestChannel(path_dir_src)
path_list_masks = getPathList(path_dir_src)

# output
path_root_output = join(path_root_mask_classification,points_per_side,channel)

# load channel taxonomy
taxononmy, prompt_input_categories, names_all, names_subparts = getTaxonomy(channel)

# define how many objects have to be found before to jump to next mask file
while True:
    nb = int(input('Minimum nb of objects within taxonomy: '))
    if 0 < nb: break
    
# where to store the temporary json files
path_dir_tmp_anns = initPath(join(path_root_output,f'tmp_anns_{nb}'))

# load or init dictionnary
path_file_dict = join(path_root_output,f'range_of_{nb}.json')
if not exists(path_file_dict):
    final_dict = {
        'taxonomy': taxononmy,
        'text_prompts': prompt_input_categories,
        'matching_info': []
    }
    writeJson(path_file_dict,final_dict)
final_dict = readJson(path_file_dict)

# start clip
print('CLIP initialization...')
clp = initCLIP()
############################################## RUN #######################################################################

for k,p in enumerate(path_list_masks):
    print(f'{k+1}/{len(path_list_masks)}')
    
    # get image id and check if not processed already 
    image_id = getId(p)
    if image_id in [getId(tmp_ann) for tmp_ann in sorted(listdir(path_dir_tmp_anns))]: continue
    
    # load image
    image = imreadRGB(getImagePath(image_id))
    
    # load masks and look for heads, to speed up classification process
    masks = decodeMasks(readJson(p))
    masks = getHeads(masks)
    count = 0
    
    # perform CLIP_classification on first {nb} heads and store tmp file
    anns = []
    path_file_tmp_anns = join(path_dir_tmp_anns,f'{image_id}.json')
    for mask in masks:
        
        # out of AoI
        if mask['AoI']==-2: 
            score = 1.
            cat = 99
            name = "background"
            
        # crossing but span over y
        elif mask['AoI']==-1:
            score = 1.
            cat = 98
            name = "wall"
            
        # crossing
        elif mask['AoI']==0:
            score = 1.
            cat = 97
            name = "cross"
            
        # within AoI
        else:
            offset = 10
            cropped = cropMask(image,mask,offset)
            score,cat,_ = computeScore(cropped, prompt_input_categories, clp)
            name = names_all[cat]
            if name not in names_subparts: count+=1
            
        tmp_ann = {
            'image_id': image_id,
            'id': mask['id'],
            'name': name,
            'cat_id': cat,
            'score': score,
            'AoI': mask['AoI']
        }
        anns.append(tmp_ann)
        
        if count >= nb: break
    writeJson(path_file_tmp_anns, anns)


# load final dict and collect existing image_id
final_dict_anns = final_dict['matching_info']
final_dict_image_id = set([ann['image_id'] for ann in final_dict_anns])

# merge all new temporary annotations into final_dict
for path_file_tmp_anns in collectPaths(path_dir_tmp_anns):
    if getId(path_file_tmp_anns) not in final_dict_image_id:
        tmp_anns = readJson(path_file_tmp_anns)
        final_dict['matching_info'] += tmp_anns
writeJson(path_file_dict,final_dict)
