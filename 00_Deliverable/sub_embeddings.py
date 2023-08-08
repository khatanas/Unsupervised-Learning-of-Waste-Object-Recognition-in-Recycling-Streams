from helper.CLIP import initCLIP
from helper.paths import collectPaths,getId
from helper.dataframes import collectEmbeddings
from helper.annotations import getAoIs,maskOverlap,filterMasks,decodeMasks,encodeMasks
from helper.tuning import readJson,writeJson
from helper.common_libraries import sys,subprocess,os
os.environ["MKL_THREADING_LAYER"] = "GNU"
############################################## CONFIG #######################################################################
path_dir_masks = sys.argv[1]
path_list_masks = collectPaths(path_dir_masks)

path_dir_output = sys.argv[2]
existing_embeddings_timestamp = [getId(path) for path in collectPaths(path_dir_output)]

# filter out, keep {batch} first elements
batch = int(sys.argv[3])
if batch>0: path_list_masks = [path for path in path_list_masks if getId(path) not in existing_embeddings_timestamp][:batch]

if len(path_list_masks)==0: sys.exit()
############################################## RUN #######################################################################
# clean SA masks of first image
clean_masks_process = subprocess.Popen(['python', 'sub_clean_masks.py', path_list_masks[0],str(0)])

CLIP_initialized = False
for k,path_file_masks in enumerate(path_list_masks):
    #print(f'{k+1}/{len(path_list_masks)}')
    
    # ensure k-th mask cleaning is done and start mask cleaning of (k+1)-th image
    if clean_masks_process.poll() is None: clean_masks_process.wait()
    if k+1<len(path_list_masks): clean_masks_process = subprocess.Popen(['python', 'sub_clean_masks.py', path_list_masks[k+1],str(k+1)])
    
    masks = decodeMasks(readJson(path_file_masks))
    # overlapping information: default values
    for k,mask in enumerate(masks):
        # masks from manual_annotations        
        mask['timestamp_id'] = getId(path_file_masks)
        if 'AoI' not in mask.keys(): 
            mask['AoI'] = 1
            mask['AoI_duo'] = -1
        mask['is_head'] = -1
        mask['head_id'] = -1
        mask['solo'] = -1
    
    # filter out masks not from AoI
    masks = getAoIs(masks)
    if len(masks)==0: continue
    
    # get overlapping lists
    are_overlapping = maskOverlap(masks)
    for stack in are_overlapping:
        head_id = stack[0]
        head = filterMasks(masks,head_id)
        head['solo'] = 1 if len(stack)==1 else 0
        head['is_head'] = 1
        head['head_id'] = head['id'] 
        
        if len(stack)==1: continue
        members_of_head = filterMasks(masks,stack[1:])
        for mask in members_of_head:
            mask['solo'] = 0 
            mask['is_head'] = 0
            mask['head_id'] = head_id
    writeJson(path_file_masks,encodeMasks(masks))
    
    if not CLIP_initialized:    
        print('CLIP initialization...')
        clp = initCLIP()
        CLIP_initialized = True
    collectEmbeddings(masks,path_dir_output,clp)
