from config_.paths import *
from config_.parameters import nb_threads,batch_size

from helper.scriptIO import setAoI,setGenerator,requestChannel,requestBool
from helper.paths import initPath,getId,collectPaths
from helper.tuning import writeJson,imreadRGB,readJson
from helper.annotations import encodeMasks,bboxLocation,getAoIs,correctMaskAoI
from helper.dataframes import mergeDf
from helper.common_libraries import os,torch,subprocess,join,exists,time

cropWidth = lambda image,duo: image[:,duo[0]:duo[1],:]
filterMasks = lambda masks: [mask for mask in masks if not (mask['bbox'][0] <= 10 or mask['bbox'][0]+mask['bbox'][2] >= hw_cropped[1]-10)]

def belongToAoI(mask):
    [[x_start,x_stop],[y_start,y_stop]] = bboxLocation(mask)
    # start and finish within range
    if duo[0] < x_start and x_stop < duo[1]: belong_to_AoI = 1
    # start after or finish before
    elif duo[1] < x_start or x_stop < duo[0]: belong_to_AoI = -2
    # cross + span over y
    elif y_start <= 10 and y_stop >= image.shape[0]-10: belong_to_AoI = -1
    # cross
    else: belong_to_AoI = 0
    return belong_to_AoI

if __name__ == "__main__":
    os.environ["MKL_THREADING_LAYER"] = "GNU"
    torch.cuda.empty_cache()
    ############################################## CONFIG #######################################################################
    # get list of images
    channel, path_dir_src = requestChannel(path_root_images_from_src)
    path_list_images = collectPaths(path_dir_src)
    
    # init/read info dict
    path_file_info = join(path_root_masks_from_images,'info_dict.json')
    if not exists(path_file_info): writeJson(path_file_info,{})
    file_info = readJson(path_file_info)
    
    if channel not in file_info.keys():
        merging = False
        # define interest area if requested
        request_AoI = requestBool('AoI')
        duo = setAoI(path_list_images) if request_AoI else -1
        process_AoI_only = False
    else:
        merging = True 
        # get duo from coarsest masks (arbitrary)
        coarsest = str(min([int(item) for item in file_info[channel].keys()]))
        duo = readJson(collectPaths(join(path_root_masks_from_images,coarsest,channel))[0])[0]['AoI_duo']
        request_AoI = type(duo)==list
        process_AoI_only = request_AoI
        
    # initialize mask generator
    mask_generator, points_per_side = setGenerator(path_list_images,duo=duo if merging else -1)
    
    # output path
    path_dir_output_masks = initPath(join(path_root_masks_from_images,f'{points_per_side}',channel))
    path_dir_output_emb = initPath(join(path_root_embeddings_from_masks,f'{points_per_side}',channel,'df_images'))
    path_dir_output_emb_merged = initPath(join(path_root_embeddings_from_masks,f'{points_per_side}',channel,'df_merged'))
    
    # save computation info
    start_time = time.time()
    ############################################## RUN #######################################################################
    #********************************* collect masks and some embeddings *****************************************************
    for k, path_file_image in enumerate(path_list_images):
        print(f'{k+1}/{len(path_list_images)}')
        
        # check if already exists
        path_file_masks = join(path_dir_output_masks,getId(path_file_image)+'.json')
        if not exists(path_file_masks):
            # read image and generate masks
            image = imreadRGB(path_file_image)
            hw0 = image.shape[:2]
            
            if process_AoI_only: 
                image = cropWidth(image,duo)
                hw_cropped = image.shape[:2]
            
            # process the image
            masks = mask_generator.generate(image)
            
            # Get rid of masks having their bbox starting or finishing too close to the AoI border, as they might be cropped masks
            if process_AoI_only: masks = filterMasks(masks)#print('dummy')
            
            # add extra info
            for mask in masks:
                # timestamp
                timestamp = getId(path_file_image) 
                mask['timestamp_id'] = timestamp
                
                # pad the binary masks if necessary and correct dimension info
                if process_AoI_only: mask = correctMaskAoI(mask,duo,hw0)
                
                # If "request_AoI" then we need to know if the mask:
                # i) is outside of AoI ==> AoI=-2
                # ii) is crossing but span over y ==> AoI=-1 (probably a wall)
                # iii) is crossing and not spanning ==> AoI=0
                # iv) belongs to AoI ==> AoI = 1
                belong_to_AoI = belongToAoI(mask) if request_AoI and not process_AoI_only else 1
                mask['AoI'] = belong_to_AoI
                mask['AoI_duo'] = duo
            
            masks = encodeMasks(masks)            
            # merge all existing masks
            if merging: 
                previous_points = max(file_info[channel].keys())
                masks = readJson(join(path_root_masks_from_images,previous_points,channel,timestamp+'.json')) + masks
            
            # save masks
            writeJson(path_file_masks, masks)
            
        # init embedding subprocess and collect for first masks
        if k==0: 
            embeddings_process = subprocess.Popen(['python', 'sub_embeddings.py', path_dir_output_masks, path_dir_output_emb, str(1)])
            embeddings_process.wait()
        # collect further
        elif k%batch_size==0 and embeddings_process.poll() is not None: 
            tic = time.time()
            embeddings_process = subprocess.Popen(['python', 'sub_embeddings.py', path_dir_output_masks, path_dir_output_emb, str(batch_size)])
            
    print('')
    print('All masks collected!')
    print('')
    
    #******************************* collect embeddings using multithreads *****************************************************
    print('Waiting to launch embeddings threads')
    embeddings_process.wait()
    
    tac = time.time()
    mean = (tac-tic)/batch_size
    
    # remaining masks to process
    path_list_masks = collectPaths(path_dir_output_masks)
    existing_embeddings_timestamp = [getId(path) for path in collectPaths(path_dir_output_emb)]
    path_list_masks_filtered = [path for path in path_list_masks if getId(path) not in existing_embeddings_timestamp]
    
    nb_files = len(path_list_masks_filtered)
    print(f'{nb_files} masks files to process using {nb_threads} threads.\n', f'Will take approximately {(nb_files*mean)/(nb_threads*3600):.2f} hours.')
    
    torch.cuda.empty_cache()
    subprocess.run(['python', 'sub_embeddings_multithreading.py', path_dir_output_masks, path_dir_output_emb, str(nb_threads)])
    
    print('')
    print('All embeddings collected!')
    print('')
    
    stop_time = time.time()
    
    print('Merging dataframes')
    [nb_masks,nb_masks_in_AoI,nb_heads_in_AoI],_ = mergeDf(path_dir_output_emb,path_dir_output_emb_merged)
    print('Done!')
    
    # ************************************************ Save run info ***********************************************************
    print('')
    total_time = int(stop_time-start_time)
    nb_images = len(path_list_images)
    print(f'Channel {channel}: required {total_time/3600:.2f} hours.\n {nb_images} images processed. {nb_masks} masks generated.')
    
    stats = {
        'computation_time':total_time,
        'nb_images':nb_images,
        'nb_masks':nb_masks,
        'nb_masks_in_AoI':nb_masks_in_AoI,
        'nb_heads_in_AoI':nb_heads_in_AoI,
        }
    
    if channel not in file_info.keys(): file_info.update({channel:{points_per_side:stats}})
    elif points_per_side not in file_info[channel].keys(): file_info[channel].update({points_per_side:stats})
    else: file_info[channel][points_per_side] = stats
    writeJson(path_file_info,file_info)