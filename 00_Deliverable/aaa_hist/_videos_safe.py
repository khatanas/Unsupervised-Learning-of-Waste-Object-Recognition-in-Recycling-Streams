from helper._paths import initDirPath, collectPaths


from os.path import join
import numpy as np
from datetime import datetime, timedelta
from copy import deepcopy
import cv2


def videoFilters(
    channels: list,
    years: list,
    months: list= [i.zfill(2) for i in np.arange(1,13).astype(str)],
    days: list=[i.zfill(2) for i in np.arange(1,32).astype(str)]):
    '''
    Creates a list of item "channel/year/month/day" of any combination of given parameters
    The months and days input are [1,...,12] and [1,...,31] by default, respectively    
    '''
    filters=[]
    for ch in channels:
        for y in years:
            for m in months:
                for d in days:
                    filters.append(join(ch,y,m,d))
    return filters


def filterVideos(videos:list, filters:list):
    '''
    Return the input list filtered with the list of given filters 
    '''
    return [v for v in videos if any(f in v for f in filters)]


def initTimestamp(path_file_video,channel):
    '''
    Create a datetime variable based on the location and name of the video file
    '''
    splitted = path_file_video.split('/')
    dd = splitted[-2]
    mm = splitted[-3]
    yyyy = splitted[-4]
    
    '.../arc/cameras/CH00001/2022/10/21/Argos_00_20221021075511.mp4'
    if channel == 'CH00001':
        splitted=path_file_video.split('_')[-1].split('.')[0]
        h = splitted[-6:-4]
        m = splitted[-4:-2]
        s = splitted[-2:]    
    
    '.../arc/cameras/CH00004/2023/03/02/2023-03-02T07%3A35%3A34.017065.avi'
    if channel == 'CH00004':
        #from file name
        splitted=path_file_video.split('%3A')
        h = splitted[0][-2:]
        m = splitted[1]
        s = splitted[-1][:2]    
        
    '.../arc/cameras/IN00001/2023/03/23/Argos_00_20230323194402.mp4'
    if channel == 'IN00001':
        splitted=path_file_video.split('_')[-1].split('.')[0]
        h = splitted[-6:-4]
        m = splitted[-4:-2]
        s = splitted[-2:]    
        
    dt = datetime.strptime(f'{yyyy}{mm}{dd}_{h}{m}{s}',"%Y%m%d_%H%M%S")
    
    return dt


def updateTimestamp(dt, seconds):
    '''
    Adds {seconds} seconds to {dt}
    '''
    return (dt+timedelta(seconds=seconds))


def stringTimestamp(dt):
    '''
    Convert a dt variable to a string representation
    '''
    return dt.strftime("%Y%m%d_%H%M%S")


def makeFilename(channel, dt):
    '''
    Creates a file name based on provided channel and datetime variable
    {channel}_yyyymmdd_hhmmss.jpg
    '''
    return f'{channel}_{stringTimestamp(dt)}.jpg'



def mse(image1, image2):
    """
    Compute mse error between image1 and image2
    """
    return np.mean((image1.astype(np.int16) - image2) ** 2)


def updateMean(image, mean_acc, count):
    mean = np.mean(image)
    mean_acc = (count*mean_acc+mean)/(count+1)
    count += 1
    return mean_acc, count


def updateFrameId(frame_id,dt,interval,fps,direction=True):
    if direction:
        frame_id += int(interval*fps)
        dt = updateTimestamp(dt,interval)
    else:
        frame_id -= int(interval*fps)
        dt = updateTimestamp(dt,-interval)
    return frame_id,dt


def extractImages(path_file_video, path_root_videos, path_root_images, threshold_factor=.5, sampling_interval=1, deadtime=5, desired_width=-1):
    '''
    Extract image samples from a video rooted at {path_root_videos} and save the images at a similar location, rooted at {path_root_images}
    The images are sampled each {interval} seconds and saved if the 2-norm between the last saved image and the new image is greater than {threshold_factor}*mean(video).
    Each time the new_image is not different enough, the sampling interval is doubled. This dynamic sampling interval is called prunning_interval.
    Once a different image is found, if this prunning interval is larger than backward_walk_threshold, the beginning of the sequence is searched in a binary-search tree fashion. 
    '''
    channel =  path_file_video.split('/')[-5]
    
    # path to saving location, and location content
    path_dir_image = initDirPath(path_file_video, path_root_videos, path_root_images)
    path_list_image = collectPaths(path_dir_image)
    
    # video information
    video_capture = cv2.VideoCapture(path_file_video)
    fps = video_capture.get(cv2.CAP_PROP_FPS)
    total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    video_duration = fps*total_frames
    
    # test video duration
    if video_duration <= 2*deadtime: print('Video too short: {path_file_video}')
    else:
        # init capture position
        frame_id = deadtime*fps
        dt = updateTimestamp(initTimestamp(path_file_video,channel),deadtime)
        frame_stop_id = total_frames-deadtime*fps
            
        # capture
        video_capture.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
        success_flag, last_saved_image = video_capture.read()
        
        # test capturable video (not corrupted)
        if not success_flag: print(f'No initial success: {path_file_video}')
        else:
            # init video mean
            mean_acc, count = updateMean(last_saved_image,0,0)
                
            # test enough light
            if mean_acc <= 25:  print(f'Night: {path_file_video}')
            else:
                #****************************** Extract **********************************************
                # get video dimension 
                hw_video = last_saved_image.shape[:2]
                
                # init fast-forward sampling
                prunning_interval=sampling_interval
                
                # set backward-walk threshold:
                # if 8 repetitons, prunning interval = 2^8 = 256*sampling_interval
                # assuming sampling interval is 1 or 2 => 256/60 or 512/60 ~= 4 to 8 minutes
                backward_walk_threshold = (2**8)*sampling_interval                           
                
                # init reference image: last saved in path_dir_image if exists, else extract from video
                if len(path_list_image)>0: 
                    last_saved_image = cv2.imread(path_list_image[-1])
                    if desired_width>0: last_saved_image = cv2.resize(last_saved_image, (hw_video[1],hw_video[0]))    
                else:
                    path_file_image = join(path_dir_image, makeFilename(channel,dt))
                    
                    to_jpg = deepcopy(last_saved_image)
                    if desired_width>0:
                        scaling_factor_x = last_saved_image.shape[1]/desired_width
                        to_jpg = cv2.resize(to_jpg, (desired_width, (int(last_saved_image.shape[0]/scaling_factor_x))))
                    cv2.imwrite(path_file_image, to_jpg)
                    frame_id,dt = updateFrameId(frame_id,dt,sampling_interval,fps)      
                    
                # capture all video
                while frame_id<frame_stop_id:
                    # capture
                    video_capture.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
                    success_flag, new_image = video_capture.read()
                    if not success_flag:
                        print(f'No further success: {path_file_video}')
                        break
                    mean_acc, count = updateMean(new_image,mean_acc,count)
                    
                    # test "different enough"
                    if np.sqrt(mse(last_saved_image,new_image))<threshold_factor*mean_acc:
                        prunning_interval*=2
                        frame_id,dt = updateFrameId(frame_id,dt,prunning_interval,fps)
                    else:
                        # reset prunning_interval
                        # if prunning_interval was too large, find beginning of sequence by iteratively jump backward/forward
                        if prunning_interval<backward_walk_threshold: prunning_interval=sampling_interval
                        else:
                            #jump backward
                            prunning_interval/=2
                            frame_id,dt = updateFrameId(frame_id,dt,prunning_interval,fps,direction=False)  
                            while prunning_interval > backward_walk_threshold:
                                # test new location
                                video_capture.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
                                success_flag, new_image = video_capture.read()
                                if not success_flag:
                                    print(f'No further success: {path_file_video}')
                                    break
                                
                                prunning_interval/=2
                                # if mse < threshold ==> too similar, go further: direction=True
                                # if mse >= threshold ==> different, jump backward: direction=False
                                frame_id,dt = updateFrameId(frame_id,dt,prunning_interval,fps, direction=np.sqrt(mse(last_saved_image,new_image))<threshold_factor*mean_acc) 
                        
                        last_saved_image = new_image
                        path_file_image = join(path_dir_image, makeFilename(channel,dt))
                        
                        to_jpg = deepcopy(last_saved_image)
                        if desired_width>0:
                            scaling_factor_x = last_saved_image.shape[1]/desired_width
                            to_jpg = cv2.resize(to_jpg, (desired_width, (int(last_saved_image.shape[0]/scaling_factor_x))))
                        cv2.imwrite(path_file_image, to_jpg)
                        frame_id,dt = updateFrameId(frame_id,dt,sampling_interval,fps)
                    
    # Release the video capture object
    video_capture.release()



