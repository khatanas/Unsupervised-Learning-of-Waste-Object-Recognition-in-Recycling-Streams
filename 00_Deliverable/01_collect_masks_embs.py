from config.paths import *
from config.parameters import nb_threads,batch_size,nb_max_rows

from helper.scriptIO import setAoI,setGenerator,requestChannel,requestBool,getPathList
from helper.paths import initPath,getId,collectPaths
from helper.tuning import writeJson, imreadRGB,readJson
from helper.annotations import encodeMasks,bboxLocation
from helper.dataframes import mergeDf
from helper.common_libraries import os,torch,subprocess,join,exists,time

os.environ["MKL_THREADING_LAYER"] = "GNU"
torch.cuda.empty_cache()
############################################## CONFIG #######################################################################
# get list of images
channel, path_dir_src = requestChannel(path_root_images_from_src)
path_list_images = getPathList(path_dir_src)

# define interest area
request_AoI = requestBool('AoI')
if request_AoI: duo = setAoI(path_list_images)

# initialize mask generator
mask_generator, points_per_side = setGenerator(path_list_images)

# output path
path_dir_output_masks = initPath(join(path_root_masks_from_images,f'{points_per_side}',channel))
path_dir_output_emb = initPath(join(path_root_embeddings_from_masks,f'{points_per_side}',channel,'df_images'))
path_dir_output_merged = initPath(join(path_root_embeddings_from_masks,f'{points_per_side}',channel,'df_merged'))

############################################## RUN #######################################################################
#********************************* collect masks and some embeddings *****************************************************
for k, path_file_image in enumerate(path_list_images):
    print(f'{k+1}/{len(path_list_images)}')
    
    # check if already exists
    path_file_masks = join(path_dir_output_masks,getId(path_file_image)+'.json')
    if not exists(path_file_masks):
        # read image and generate masks
        image = imreadRGB(path_file_image)
        masks = mask_generator.generate(image)
        
        # add extra info
        for mask in masks:
            # timestamp
            mask['timestamp_id'] = getId(path_file_image) 
            
            # AoI    
            if request_AoI: 
                [[x_start,x_stop],[y_start,y_stop]] = bboxLocation(mask)
                # start and finish within range
                if duo[0] < x_start and x_stop < duo[1]: belong_to_AoI = 1
                # start after or finish before
                elif duo[1] < x_start or x_stop < duo[0]: belong_to_AoI = -2
                # cross + span over y
                elif y_start <= 10 and y_stop >= image.shape[0]-10: belong_to_AoI = -1
                # cross
                else: belong_to_AoI = 0
            else: belong_to_AoI = 1
            mask['AoI'] = belong_to_AoI
            mask['AoI_duo'] = duo if request_AoI else -1
        # save
        writeJson(path_file_masks, encodeMasks(masks))
            
    # init embedding subprocess and collect for first masks
    if k==0: 
        tic = time.time()
        embeddings_process = subprocess.Popen(['python', 'sub_embeddings.py', path_dir_output_masks, path_dir_output_emb, str(1)])
        embeddings_process.wait()
        tac = time.time()
        mean = tac-tic
    # collect further
    elif k%batch_size==0 and embeddings_process.poll() is not None: 
        tic = time.time()
        embeddings_process = subprocess.Popen(['python', 'sub_embeddings.py', path_dir_output_masks, path_dir_output_emb, str(batch_size)])

print('')
print('All masks collected!')
print('')

#******************************* collect embeddings using multithreads *****************************************************
print('Waiting to launch embeddings multithreads')
embeddings_process.wait()

tac = time.time()
mean += tac-tic
mean /= batch_size+1

# remaining masks to process
path_list_masks = collectPaths(path_dir_output_masks)
existing_embeddings_timestamp = [getId(path) for path in collectPaths(path_dir_output_emb)]
path_list_masks_filtered = [path for path in path_list_masks if getId(path) not in existing_embeddings_timestamp]

nb_files = len(path_list_masks_filtered)
print(f'{nb_files} masks files to process using {nb_threads} threads.\n', f'Will take approximately {(nb_files*mean)/(nb_threads*3600):.2f} hours.')

torch.cuda.empty_cache()
subprocess.run(['python', 'sub_embeddings_multithreading.py', path_dir_output_masks, path_dir_output_emb, str(nb_threads)])
torch.cuda.empty_cache()

print('')
print('All embeddings collected!')
print('')

print('Merging dataframes')
mergeDf(path_dir_output_emb,path_dir_output_merged)

print('Done!')