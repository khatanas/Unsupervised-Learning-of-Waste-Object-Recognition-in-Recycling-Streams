from config_.paths import *

from helper.annotations import encodeMasks,decodeMasks,filterMasks,matchingAnnotations
from helper.coco import getAnnsFromImageId, getCocoImageId
from helper.scriptIO import requestChannel, requestPtsPerSide
from helper.tuning import readJson, writeJson,sortedUnique
from helper.dataframes import readDf, writeDf, extractValues
from helper.paths import getId,initPath
from helper.common_libraries import np,shutil

def updateRows(points,channel,timestamp,df):
    """
    Extracts GT information from masks file and insert it in provided df 
    """
    path_file_masks = join(path_root_masks_from_images,points,channel,timestamp+'.json')
    masks = readJson(path_file_masks)
    
    info = [[] for _ in range(3)]
    for mask in masks:
        info[0].append(mask['id'])
        info[1].append(mask['GT_id'])
        info[2].append(mask['GT_category'])
        
    for sa, gt, cat in zip(info[0],info[1],info[2]):
        mask = np.logical_and(df['timestamp_id']==timestamp, df['SA_id']==sa)
        df.loc[mask, ['in_GT','GT_id','GT_category']] = 1,gt,cat
    return df

def updateMerged(points,channel,list_timestamps_GT):
    """
    Updates all merged file by including the GT information contained in the masks files
    """
    # load merged dict 
    path_file_merged_dict = join(path_root_embeddings_from_masks,points,channel,'merged_dict.json')
    merged_dict = readJson(path_file_merged_dict)
    
    # if no mask are extracted for an image using SA (it happens scarsely), then the corresponding df is not created (because the json file is empty)
    # and its name does not appear in merged_dict (as it does not exist, it has never been merged).
    list_timestamps_GT = [item for item in list_timestamps_GT if item in merged_dict.keys()]
    
    # get the ids of the merged dict to update. 
    merged_ids = sortedUnique([merged_dict[item] for item in list_timestamps_GT])
    
    # iterate over the required merged files
    path_dir_merged = join(path_root_embeddings_from_masks,points,channel,'df_merged')
    for merged_id in merged_ids:
        # load df
        path = join(path_dir_merged, f'merged{merged_id}.csv')
        df = readDf(path)
        
        # rows from merged file related to GT 
        candidate_rows = df[df['timestamp_id'].isin(list_timestamps_GT)]
        # collect timestamps
        filtered_timestamp = sortedUnique(extractValues(candidate_rows,'timestamp_id'))
        # for each of them, update the rows (change in_GT, GT_id, GT_category)
        for timestamp in filtered_timestamp: 
            df = updateRows(points,channel,timestamp,df)   
        # save updated
        writeDf(path,df)

if __name__ == "__main__":
    points, path_dir_src = requestPtsPerSide(path_root_embeddings_from_masks)
    channel, path_dir_src = requestChannel(path_dir_src)
    
    # GT_file
    path_file_GT = join(path_root_GT, channel, f'{channel}_coco_annotations.json')
    coco_annotations = readJson(path_file_GT)
    
    # timestamp info
    list_timestamps_GT = [getId(item['file_name']) for item in coco_annotations['images']]
    
    print('Updating masks files...')
    # for each timestamp, update json file:
    for k,timestamp in enumerate(list_timestamps_GT):
        if k%100 ==0: print(f'{k}/{len(list_timestamps_GT)}')
        
        # get SA masks
        path_file_masks = join(path_root_masks_from_images,points,channel,timestamp+'.json')
        masks = decodeMasks(readJson(path_file_masks))
        
        # get GT masks
        image_id = getCocoImageId(timestamp,coco_annotations)
        masks_GT = decodeMasks(getAnnsFromImageId(image_id,coco_annotations))
        
        # find the matching ids
        is_a_match,_,_ = matchingAnnotations(masks_GT,masks)
        SA_id = [pair[1] for pair in is_a_match]
        GT_id = [pair[0] for pair in is_a_match]
        GT_category = [item['category_id'] for item in filterMasks(masks_GT,GT_id)]
        
        # for each found id, store the GT information in the json file
        for mask in masks:
            mask['in_GT'] = 1
            if mask['id'] in SA_id:
                idx = SA_id.index(mask['id'])
                mask['GT_id'] = GT_id[idx]
                mask['GT_category'] = GT_category[idx]
        # save updates
        writeJson(path_file_masks,encodeMasks(masks))
    
    # once all the json files are updated, update the merged files accordingly
    print('Updating merged file...')
    updateMerged(points,channel,list_timestamps_GT)
    
    '''# create a backup for the masks and the embeddings
    print('Creating backup files...')
    for path_root in [path_root_masks_from_images, path_root_embeddings_from_masks]:
        path_dir_backup = initPath(join(path_root,points,'00_backup'))
        path_dir_files = join(path_root, points, channel)
        shutil.copytree(path_dir_files, join(path_dir_backup,channel))'''
    
    print('Done!')