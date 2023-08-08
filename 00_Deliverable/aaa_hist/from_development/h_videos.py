import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from os import listdir, makedirs
from os.path import join, exists, isfile, getsize
from datetime import datetime, timedelta


def collectFiles(roots:list, excluded = ['txt']):
    '''
    Return a list of all files belonging to any path whose root is in roots input
    The extension in excluded are not kept 
    '''
    for item in roots:
        sub = [] if isfile(item) else listdir(item) 
        roots += [join(item,elem) for elem in sub]
    return [item for item in roots if isfile(item) and not any([item.endswith(extension) for extension in excluded])]


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


def makePath(path2file,root_videos,root_imgs):
    '''
    Returns the location of the file, creates the directory if not existing
    '''
    tmp = path2file.split('/')
    file_name = tmp[-1]
    loc = path2file.replace(root_videos,root_imgs)
    loc = path2file.replace(f'{file_name}','')
    
    if not exists(loc): makedirs(loc)
    
    return loc


def getImageId(string):
    '''
    The 23 first character of the file name creates a unique id 
    7 characters for location. Ex: CH00004
    8 characters for day. Ex: 20230401
    6 characters for time, Ex: 235609
    2 underscores to separate last_saved_image characteristic: CH00004_20230401_235609
    '''
    return string.split('/')[-1][:23]


def initTimestamp(path2file):
    '''
    Create a datetime variable based on the location and name of the video file
    '''
    # from loc
    splitted = path2file.split('/')
    dd = splitted[-2]
    mm = splitted[-3]
    yyyy = splitted[-4]
    #from file name
    splitted=path2file.split('%3A')
    h = splitted[0][-2:]
    m = splitted[1]
    s = splitted[-1][:2]    
    
    dt = datetime.strptime(f'{yyyy}{mm}{dd}_{h}{m}{s}',"%Y%m%d_%H%M%S")
    
    return dt

def updateTimestamp(dt, seconds):
    '''
    Adds {seconds} seconds to {dt}
    '''
    return (dt+timedelta(seconds=seconds))

def strTimestamp(dt):
    '''
    Print datetime as string
    '''
    return dt.strftime("%Y%m%d_%H%M%S")

def makeFilename(channel, dt):
    '''
    Creates a file name based on provided channel and datetime variable
    {channel}_yyyymmdd_hhmmss.jpg
    '''
    return f'{channel}_{strTimestamp(dt)}.jpg'


def mse(image1, image2):
    """
    Compute mse error between image1 and image2
    """
    return np.mean((image1.astype(np.int16) - image2) ** 2)


def mae(image1, image2):
    """
    Compute mae error between image1 and image2
    """
    return np.mean(abs(image1.astype(np.int16) - image2))

def extractImages(path2file,root_videos,root_imgs,threshold=35,interval=5):
    '''
    Extract all images from a video with an interval of {interval} seconds 
    Current frame is extracted if mse(last_extracted_frame, current_frame)>threshold
    '''
    output_path = makePath(path2file,root_videos,root_imgs)
    channel =  path2file.split('/')[-5]
    dt = initTimestamp(path2file)
    
    # video information
    video_capture = cv2.VideoCapture(path2file)
    fps = video_capture.get(cv2.CAP_PROP_FPS)
    total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    new_frame_pos = 0
    
    # init "last_extracted" as initial frame
    success_flag, previous = video_capture.read()
    image_path = join(output_path,makeFilename(channel,dt))
    if success_flag: cv2.imwrite(image_path, previous)
    
    while video_capture.isOpened():
        # read 1 frame each {interval} second of video instead of all
        new_frame_pos += min(total_frames - 1, int(interval*fps))
        video_capture.set(cv2.CAP_PROP_POS_FRAMES, new_frame_pos)
        success_flag, frame = video_capture.read()
        if not success_flag: break

        if mse(previous,frame)>threshold:
            #print(mse(previous,rgb_frame))
            previous = frame
            image_path = join(output_path,makeFilename(channel,dt))
            cv2.imwrite(image_path, frame)
            
        dt = updateTimestamp(dt,interval)

    # Release the video capture object
    video_capture.release()