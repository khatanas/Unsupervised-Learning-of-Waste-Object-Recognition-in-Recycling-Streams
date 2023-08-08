from config.paths import *
from config.parameters import max_width

from helper.videos import extractImages
from helper.scriptIO import requestChannel,requestBool,requestInt,requestFloat,requestSrcPath,getPathList
from helper.tuning import writeJson
from helper.paths import initPath,alterPath
from helper.common_libraries import Image,listdir,exists

if requestBool('From video'): ############# Video processing ####################################################################
    # ****************************************** CONFIG ************************************************************************
    # get list of videos
    channel,path_dir_src = requestChannel(path_root_videos)
    path_list_videos = getPathList(path_dir_src)
    
    # extraction parameters
    similarity_threshold = requestFloat('Similarity threshold',a_included=False,b_included=False)
    sampling_interval = requestInt('Sampling interval, in seconds', a=1)
    
    # ******************************************* RUN **************************************************************************
    for idx,path in enumerate(path_list_videos):
        print(f'processing video {idx+1}/{len(path_list_videos)}')
        path_dir_output = alterPath(path_root_videos, path_root_images_from_src, path)
        extractImages(path, path_dir_output, similarity_threshold, sampling_interval)
        
else: ###########################   Load images from some src, format name #####################################################
    # get source
    path_dir_src = requestSrcPath()
    
    # request new channel name (required to respect pipeline format)
    while True:
        print('Please enter 5 digits: XX . . . . .')
        new_channel_digits = input('New channel name: XX')
        new_channel = 'XX'+new_channel_digits
        if len(new_channel) == 7 and not exists(join(path_root_images_from_src,new_channel)):break
    
    # assign dummy year/month/day path (required to respect pipeline format) (https://www.globalrecyclingday.com/)
    path_dir_dest = initPath(join(path_root_images_from_src,new_channel,'2023','03','18'))
    
    # init dictionary for path correspondences
    dict_channel = {
        'root':{path_dir_src: new_channel},
        'file_name':{}
    }
    
    # copy images and format name
    print('Copying images...')
    count=0
    for k,file_name in enumerate(sorted(listdir(path_dir_src))[:999999]):
        print(f'{k+1}/{len(listdir(path_dir_src))}')
        path_file_image_src = join(path_dir_src,file_name)
        
        # to RGB
        image = Image.open(path_file_image_src)
        if image.mode != "RGB":image = image.convert("RGB")
        
        # resize if necessary
        width, height = image.size
        if width>max_width: 
            scaling_factor_x = width/max_width
            image = image.resize((width,int(height/scaling_factor_x)))
        
        # format name
        new_name = f'{new_channel}_20230318_{str(count).zfill(6)}.jpg'
        path_file_image_dest = join(path_dir_dest,new_name)
        
        # save image
        dict_channel['file_name'][file_name]= path_file_image_dest.split(f'{new_channel}/')[-1]
        image.save(path_file_image_dest)
        count+=1
        
    # save dictionary in channel directory
    writeJson(join(path_root_images_from_src,f'{new_channel}.json'),dict_channel)