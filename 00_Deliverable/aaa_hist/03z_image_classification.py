from helper._config_paths import *
from helper._config_classification import getTaxonomy, channel_reference_ids
from helper._scriptIO import *
from helper._tuning import readJson,writeJson
from helper._paths import getMasksPath, getImagePath,initPath
from helper._annotations import getAoIs

import shutil
############################################## CONFIG #######################################################################

# get mask_classification file
points_per_side,path_dir_src = requestPtsPerSide(path_root_mask_classification)
channel,path_dir_src = requestChannel(path_dir_src)
nb, path_dir_src = requestRange(path_dir_src)
dict_mask_classification = readJson(path_dir_src)
_,_,names_all,names_subparts = getTaxonomy(channel)


# get confidence threshold
while True:
    print('Please enter a value within [0,1]')
    classification_threshold = float(input('Classification threshold: '))
    if 0 <= classification_threshold and classification_threshold <= 1: break
path_root_output = join(path_root_image_classification, points_per_side, channel,f'{classification_threshold}')


# get sparsity_threshold
sparse_id = channel_reference_ids[channel]['sparse']
sparse_masks = readJson(getMasksPath(sparse_id,points_per_side))
AoIs = getAoIs(sparse_masks)
sparse_threshold = len(AoIs)

############################################## RUN #######################################################################

# exctract for main categories
for cat_id, object_name in enumerate(names_all):
    if object_name in names_subparts:continue
    print(f'Processing {cat_id}: {object_name}')
    
    # collecting ids...
    image_ids= [ann['image_id'] for ann in dict_mask_classification['matching_info'] 
                if ann['cat_id'] == cat_id 
                and ann['score']>=classification_threshold]

    if len(image_ids)==0: print(f'{object_name}: no instance found')
    else:
        filled_name = object_name.replace(' ','_')
        path_dir_selected_images = initPath(join(path_root_output,f'{filled_name}_{classification_threshold}'))
        
        # init dicitonnary 
        path_file_final_dict = join(path_root_output, f'{filled_name}_{classification_threshold}.json')
        final_dict = {}
        
        # counting ids...
        annotations_ids = [ann['id'] for ann in dict_mask_classification['matching_info'] 
                        if ann['cat_id'] == cat_id 
                        and ann['score']>=classification_threshold]
        ref_id = image_ids[0]
        related_ids = [annotations_ids[0]]
        count = 1
        for k in range(1,len(image_ids)):
            new_id = image_ids[k]
            if ref_id == new_id:
                count += 1    
                related_ids.append(annotations_ids[k])
            else:
                final_dict[ref_id] = {'count': count, 'ids': related_ids}
                count = 1
                related_ids = [annotations_ids[k]]
                ref_id = new_id
        final_dict[ref_id] = {'count': count, 'ids': related_ids}
        
        # analyzing sparsity...
        for image_id in final_dict:
            path_file_masks = getMasksPath(image_id, points_per_side)
            is_sparse = int(len(readJson(path_file_masks)) <= sparse_threshold)
            final_dict[image_id]['sparse'] = is_sparse
        writeJson(path_file_final_dict,final_dict)
        
        # copying images...
        for k,image_id in enumerate(final_dict):
            if k%1000 == 0: print(f'{k}/{len(final_dict)}')
            
            path_file_image = getImagePath(image_id)
            
            sparse = final_dict[image_id]['sparse']
            count = final_dict[image_id]['count']
            path_file_annotated_image = join(path_dir_selected_images,f'{sparse}_{count}_{image_id}.jpg')
            
            shutil.copy(path_file_image,path_file_annotated_image)