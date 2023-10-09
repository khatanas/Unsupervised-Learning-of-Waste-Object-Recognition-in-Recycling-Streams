from config_.paths import path_root_videos
from config_.parameters import missed,max_width,min_light,deadtime

from helper.paths import collectPaths
from helper.common_libraries import cv2,np,datetime,timedelta,deepcopy,join

#**************************************************************************************************************
# computes the mean square error
mse = lambda image1,image2: np.mean((image1.astype(np.int16) - image2) ** 2)

# some explicit lambda functions to manipulate and use the datetime variable
updateTimestamp = lambda dt,seconds: dt+timedelta(seconds=seconds)
stringTimestamp = lambda dt: dt.strftime("%Y%m%d_%H%M%S")
makeFilename = lambda channel, dt: f'{channel}_{stringTimestamp(dt)}.jpg'

def validFormat(path_file_video):
    """
    Checks if the name of the file corresponds to a valid format
    Returns a boolean True:valid/False:not valid
    """
    # path_root_videos/channel/year/month/day/file_name.ext 
    tail = path_file_video.replace(path_root_videos+'/','')
    splitted = tail.split('/')
    channel = splitted[0]
    name = splitted[-1]
    
    # [channel,year,month,day,file_name.ext]
    if len(splitted)==5:
        if channel in ['CH00001','IN00001','IN00002']:
            # CH00001: must be similar to CH00001/2022/10/21/Argos_00_20221021075511.mp4
            # IN00001: must be similar to IN00001/2023/03/23/Argos_00_20230323194402.mp4
            # IN00002: must be similar to IN00002/2022/10/27/Apollo_00_20221027070256.mp4
            splitted = name.split('_')
            if len(splitted) == 3:
                timecode = splitted[-1].split('.')[0]
                return True if len(timecode)==14 else False
        if channel == 'CH00004': 
            #CH00004: must be similar to CH00004/2023/03/02/2023-03-02T07%3A35%3A34.017065.avi'
            return True if len(name.split('.')[0])==23 else False
    return False


def initTimestamp(path_file_video):
    '''
    Creates a datetime variable based on the location and name of the video file
    '''
    splitted = path_file_video.split('/')
    dd = splitted[-2]
    mm = splitted[-3]
    yyyy = splitted[-4]
    channel = splitted[-5]
    
    if channel in ['CH00001','IN00001,IN00002']:
        #'.../arc/cameras/CH00001/2022/10/21/Argos_00_20221021075511.mp4'
        #'.../arc/cameras/IN00001/2023/03/23/Argos_00_20230323194402.mp4'
        #'...arc/cameras/IN00002/2022/10/27/Apollo_00_20221027070256.mp4'
        splitted=path_file_video.split('_')[-1].split('.')[0]
        h = splitted[-6:-4]
        m = splitted[-4:-2]
        s = splitted[-2:]    
    
    if channel == 'CH00004':
        #'.../arc/cameras/CH00004/2023/03/02/2023-03-02T07%3A35%3A34.017065.avi'
        splitted=path_file_video.split('%3A')
        h = splitted[0][-2:]
        m = splitted[1]
        s = splitted[-1][:2]    
        
    dt = datetime.strptime(f'{yyyy}{mm}{dd}_{h}{m}{s}',"%Y%m%d_%H%M%S")
    
    return dt


def updateMean(image, mean_acc, count):
    """
    Updates the dynamical mean: new_mean = (k*mean+value)/(k+1)
    """
    value = np.mean(image)
    mean_acc = (count*mean_acc+value)/(count+1)
    count += 1
    return mean_acc, count


def updateFrameId(frame_id,dt,interval,fps,direction=True):
    """
    Updates by {interval*fps} the frame_id and its associated datetime variable
    If direction is True, updates towards the end of the video
    If direction is False, updates towards the beginning of the video
    """
    if direction:
        frame_id += int(interval*fps)
        dt = updateTimestamp(dt,interval)
    else:
        frame_id -= int(interval*fps)
        dt = updateTimestamp(dt,-interval)
    return frame_id,dt


def extractImages(path_file_video, path_dir_output, similarity_threshold, sampling_interval):
    """
    Extraction algorithm, further described in the rapport
    """
    # the name of the file must respect the required format
    if not validFormat(path_file_video):
        print(path_file_video + ': incorrect format')
        return
    
    # get the channel and collect alredy existing images 
    channel =  path_file_video.split('/')[-5]
    path_list_images = collectPaths(path_dir_output)
    
    # video information
    video_capture = cv2.VideoCapture(path_file_video)
    fps = video_capture.get(cv2.CAP_PROP_FPS)
    total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    video_duration = fps*total_frames
    frame_stop_id = total_frames-deadtime*fps
    
    # ************************************ Tests *******************************************************
    # test video duration
    if video_duration <= 2*deadtime: print(f'Not long enough: {path_file_video}')
    else:
        # init capture position + initial capture
        frame_id, dt = updateFrameId(0,initTimestamp(path_file_video),deadtime,fps)
        video_capture.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
        success_flag, initial_capture = video_capture.read()
        
        # test capturable video (not corrupted)
        if not success_flag: print(f'Not capturable: {path_file_video}')
        else:
            # init video mean
            mean_acc, count = updateMean(initial_capture,0,0)
            
            # test enough light
            if mean_acc <= min_light:  print(f'Not bright enough: {path_file_video}')
            else:
                #****************************** Extract **********************************************
                # get video dimension 
                hw_video = initial_capture.shape[:2]
                height = hw_video[0]
                width = hw_video[1]
                
                # similarity condition
                is_similar = lambda ref,new: mse(ref,new) < (similarity_threshold*mean_acc)**2
                
                # init reference image
                if len(path_list_images)==0:
                    # path_dir_output empty ? ==> save first image + update frame_id
                    to_jpg = deepcopy(initial_capture)
                    if max_width>0:
                        # downscale to save
                        scaling_factor_x = width/max_width
                        to_jpg = cv2.resize(to_jpg, (max_width, (int(height/scaling_factor_x))),interpolation=cv2.INTER_LANCZOS4)
                    cv2.imwrite(join(path_dir_output, makeFilename(channel,dt)), to_jpg)
                    last_saved_image=initial_capture
                
                else: 
                    # path_dir_output not empty ? ==> use last image of the folder as reference. no frame_id update
                    last_saved_image = cv2.imread(path_list_images[-1])
                    # upscale to use as reference
                    if max_width>0: last_saved_image = cv2.resize(last_saved_image, (hw_video[1],hw_video[0]))    
                
                
                # init fast-forward sampling
                prunning_interval=sampling_interval
                # set backward-walk threshold: if 8 repetitons, prunning interval = 2^8 = 256*sampling_interval. Assuming sampling interval is 1 or 2 => 256/60 or 512/60 ~= 4 to 8 minutes
                backward_walk_threshold = (2**missed)*sampling_interval                           
                while True:
                    # update capture position and check range
                    frame_id,dt = updateFrameId(frame_id,dt,prunning_interval,fps)
                    if frame_id >= frame_stop_id: break
                    
                    # capture and update mean
                    video_capture.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
                    success_flag, new_image = video_capture.read()
                    if success_flag: mean_acc, count = updateMean(new_image,mean_acc,count)
                    else:
                        print(f'No further success: {path_file_video}')
                        break
                    
                    # test similarity
                    if is_similar(last_saved_image,new_image): prunning_interval*=2
                    else:
                        # ****************** backwards-walk **********************
                        exit_outer = False
                        found_image = deepcopy(new_image)
                        while prunning_interval>backward_walk_threshold:
                            prunning_interval/=2
                            
                            # test new location
                            frame_id,dt = updateFrameId(frame_id,dt,prunning_interval,fps,direction=is_similar(last_saved_image,found_image))  
                            video_capture.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
                            success_flag, found_image = video_capture.read()
                            if success_flag: mean_acc, count = updateMean(found_image,mean_acc,count)
                            else:
                                print(f'No further success: {path_file_video}')
                                exit_outer = True
                                break
                            
                            # reached limit
                            if prunning_interval==backward_walk_threshold:
                                # no new image found => back to last point
                                if is_similar(last_saved_image,found_image): frame_id,dt = updateFrameId(frame_id,dt,prunning_interval,fps)
                                # found new => update
                                else: new_image = found_image
                        if exit_outer: break
                        # *******************************************************
                        # save
                        to_jpg = deepcopy(new_image)
                        if max_width>0:
                            scaling_factor_x = width/max_width
                            to_jpg = cv2.resize(to_jpg, (max_width, (int(height/scaling_factor_x))),interpolation=cv2.INTER_LANCZOS4)
                        cv2.imwrite(join(path_dir_output, makeFilename(channel,dt)), to_jpg)
                        
                        last_saved_image = new_image
                        prunning_interval=sampling_interval
                        
    # Release the video capture object
    video_capture.release()