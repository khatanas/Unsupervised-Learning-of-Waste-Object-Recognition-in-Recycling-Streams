from helper._config_paths import *
from helper._scriptIO import *
from helper._lib_SA import *
from helper._paths import alterPath, getId, getImagePath, getMasksPath
from helper._tuning import readJson, writeJson,imreadRGB

import random
from os.path import join
getRandomElement = lambda collection: random.sample(collection,1)[0]
cropWidth = lambda image,duo: image[:,duo[0]:duo[1],:]

############################################## CONFIG #######################################################################
if requestTrueFalse('Use classification file'):
    # get path of classified image file
    points_per_side_sparse,path_dir_src = requestPtsPerSide(path_root_image_classification)
    channel,path_dir_src = requestChannel(path_dir_src)
    classification_threshold, path_dir_src = requestThreshold(path_dir_src)
    path_file_objects = requestObject(path_dir_src,channel,classification_threshold)

    # print info about classification file
    image_classification_info = readJson(path_file_objects)
    sparse_levels = 2
    density_levels = 5
    image_subsets_id =[[[] for _ in range(density_levels)] for _ in range(sparse_levels)]
    for sparse in [0,1]:
        for count in np.arange(5):
            [image_subsets_id[sparse][count].append(key) for key in image_classification_info.keys() 
                                                    if image_classification_info[key]['sparse']==sparse 
                                                    and image_classification_info[key]['count']==count+1]
            print(f'sparse: {sparse}, count {count+1}: {len(image_subsets_id[sparse][count])}')
    print(f'Total: {len(image_classification_info)}')

    # get subset of classified images if necessary
    if requestTrueFalse('Subset'):
        image_ids = []
        # How many images
        while True:
            nb_subset = int(input("Nb images in subset: "))
            if nb_subset<= len(image_classification_info): break
            
        # for balanced set, pick nb_subset/nb_categories elements per categories
        # but some categories don't have that many elements, so we collect more from the others
        nb_per_cat = int(nb_subset/(sparse_levels*density_levels))
        k = 0
        while len(image_ids)<nb_subset:
            result = []
            for s in range(sparse_levels):
                for d in range(density_levels):
                    result.extend(image_subsets_id[s][d][k*nb_per_cat:(k+1)*nb_per_cat])
            image_ids += result
            k+=1
        image_ids = image_ids[:nb_subset]
    else: image_ids = [key for key in image_classification_info]
    path_list_classified_images = [getImagePath(id) for id in image_ids]
    
else: 
    points_per_side_sparse,path_dir_src = requestPtsPerSide(path_root_masks_from_images)
    channel,path_dir_src = requestChannel(path_dir_src)
    path_list_classified_images = collectPaths(join(path_root_images_from_videos,channel))


# define interest area
request_AoI = requestTrueFalse('AoI')
if request_AoI:
    key = 'n'
    while True:
        image = imreadRGB(getRandomElement(path_list_classified_images))
        if key == 'y': break
        print("Define an area of interest: ")
        duo = getDuo(image)
        
        print("Validate area of interest [y/n]: ")
        image[:,duo[0]:duo[1],:] = 0
        key = imagePreview(image)
        if key == 'n': duo.clear()


# choose annotation density
key = 'n'
while True:
    if key == 'y': break
    
    while True:
        points_per_side = int(input('Points per side (default=32): '))
        if 0<points_per_side and points_per_side<=32: break
        
    while True:
        points_per_batch = int(input('Points per batch (default=64): '))
        if 0<points_per_batch and points_per_batch<=64: break
        
    print('Please wait during preview generation...')    
    mask_generator = SamAutomaticMaskGenerator(
        model = initSam(),
        points_per_side=points_per_side,
        points_per_batch=points_per_batch,
        )
    image = imreadRGB(getRandomElement(path_list_classified_images))
    cropped_image = cropWidth(image,duo) if request_AoI else image
    
    masks = mask_generator.generate(cropped_image)
    print("Validate mask density [y/n]: ")
    key = imagePreview(cropped_image,masks)

############################################## RUN #######################################################################

#collect fine masks
hw_full = image.shape[:2]
for image_idx, path_file_image in enumerate(path_list_classified_images):
    torch.cuda.empty_cache()
    print(f'{image_idx+1}/{len(path_list_classified_images)}')
    
    # read image and crop it to compute fine SAM on AoI only (if AoI defined)
    image = imreadRGB(path_file_image)
    if request_AoI:
        image = cropWidth(image,duo)
        hw_cropped = image.shape[:2]
    masks = mask_generator.generate(image)
    
    # remove masks beginning or ending to close from AoI boundary (likely to cross)
    if request_AoI: masks = [mask for mask in masks if not (mask['bbox'][0] <= 10 or mask['bbox'][0]+mask['bbox'][2] >= hw_cropped[1]-10)]
    
    # add id, AoI info and pad segmentation mask 
    for k,mask in enumerate(masks):
        if request_AoI:
            # pad segmentation
            to_pad = mask['segmentation']
            padded = np.pad(to_pad,((0,0),(duo[0],hw_full[1]-duo[1])),mode='constant',constant_values=False)
            assert(padded.shape[:2])==hw_full
            mask['segmentation'] = padded
            
            # update bbox and prompt point
            mask['bbox'][0] += duo[0]
            mask['point_coords'][0][0] += duo[0]
        
        # add id
        mask['id'] = k
        mask['AoI'] = 1
    
    # save
    masks = encodeMasks(masks)
    _ = alterPath(path_root_images_from_videos,join(path_root_fine_masks,f'{points_per_side}'),path_file_image)
    writeJson(getMasksPath(getId(path_file_image),points_per_side,sub=True), masks)


# merge with sparse_crossing_masks
path_list_masks = collectPaths(join(path_root_fine_masks,f'{points_per_side}',channel))
for p in path_list_masks:
    fine_masks = readJson(p)
    sparse_masks = readJson(getMasksPath(getId(p), points_per_side_sparse))
    sparse_crossing_masks = [mask for mask in sparse_masks if mask['AoI']==0]
    merged = fine_masks + sparse_crossing_masks
    for k,mask in enumerate(merged):
        mask['id'] = k
    writeJson(p,merged)

'''
# copy images to visualize
path_dir_images_subsets_train = join(path_root_image_subsets,channel)
os.makedirs(path_dir_images_subsets_train,exist_ok=True)
for path_file_image in path_list_classified_images:
    shutil.copy(path_file_image,join(path_dir_images_subsets_train, getName(path_file_image)))'''
