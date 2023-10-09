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

batch = int(sys.argv[3])
# if batch <= 0, the masks from path_dir_masks are processed and the embeddings are stored in path_dir_output (overwritten if already exist)
# if batch > 0, the {batch} first masks from path_dir_masks, whose embedding don't exists in path_dir_ouptut are processed
if batch>0: path_list_masks = [path for path in path_list_masks if getId(path) not in existing_embeddings_timestamp][:batch]

# exit if mask directory is empty
if not path_list_masks: sys.exit()
############################################## RUN #######################################################################

CLIP_initialized = False
for k,path_file_masks in enumerate(path_list_masks):    
    if k==0: clean_masks_process = subprocess.Popen(['python', 'sub_clean_masks.py', path_list_masks[0]])
    
    # ensure k-th mask cleaning is done and start mask cleaning of (k+1)-th image
    clean_masks_process.wait()
    if (k+1)<len(path_list_masks): clean_masks_process = subprocess.Popen(['python', 'sub_clean_masks.py', path_list_masks[(k+1)]])
    
    # read and decode cleaned masks
    masks = decodeMasks(readJson(path_file_masks))
    
    # add overlap information
    are_overlapping = maskOverlap(masks)
    for stack in are_overlapping:
        head_id = stack[0]
        head = masks[head_id]
        head['solo'] = 1 if len(stack)==1 else 0
        head['is_head'] = 1
        head['head_id'] = head['id'] 
        
        if len(stack)==1: continue
        members_of_head = filterMasks(masks,stack[1:])
        for mask in members_of_head:
            mask['solo'] = 0 
            mask['is_head'] = 0
            mask['head_id'] = head_id
    
    # add id info, and AoI default value if necessary
    for k,mask in enumerate(masks):
        mask['timestamp_id'] = getId(path_file_masks)
        mask['in_GT'] = 0
        mask['GT_id'] = -1
        mask['GT_category'] = -1
        if 'AoI' not in mask.keys(): 
            mask['AoI'] = 1
            mask['AoI_duo'] = -1
    
    # save updated file
    writeJson(path_file_masks,encodeMasks(masks))
    
    # filter out masks not entirely from AoI or crossing
    masks = getAoIs(masks)
    if not CLIP_initialized and len(masks)>0:    
        print('CLIP initialization...')
        clp = initCLIP()
        CLIP_initialized = True
        
    if CLIP_initialized and len(masks)>0: collectEmbeddings(masks,path_dir_output,clp)
