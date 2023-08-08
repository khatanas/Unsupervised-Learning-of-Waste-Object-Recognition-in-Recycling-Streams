from helper._config_paths import *
from helper._config_classification import *
from helper._scriptIO import *
from helper._CLIP import *
from helper._tuning import readJson,imreadRGB
from helper._annotations import decodeMasks, cropMask
from helper._paths import getImagePath,getId,initPath,getName
from helper._df import *

import subprocess
############################################## CONFIG #######################################################################

# get list of masks
points_per_side, path_dir_src = requestPtsPerSide(path_root_fine_masks)
channel, path_dir_src = requestChannel(path_dir_src)
path_list_masks = collectPaths(path_dir_src)

taxononmy, prompt_input_categories, names_all, names_subparts = getTaxonomy(channel)

# output
path_dir_output = initPath(join(path_root_embeddings,points_per_side,channel,'df_images'))
############################################## RUN #######################################################################
# clean SA masks of first image
clean_masks_process = subprocess.Popen(['python', 'sub_clean_masks.py', path_list_masks[0]])

# start clip
print('CLIP initialization...')
clp = initCLIP()

for k,path_file_masks in enumerate(path_list_masks):
    print(f'{k+1}/{len(path_list_masks)}')
    
    # load image
    path_file_image = getImagePath(getId(path_file_masks))
    timestamp_id = getName(path_file_image)
    image = imreadRGB(path_file_image)
    
    # ensure k-th mask cleaning is done and start mask cleaning of (k+1)-th image
    if clean_masks_process.poll() is None: clean_masks_process.wait()
    if k+1<len(path_list_masks): clean_masks_process = subprocess.Popen(['python', 'sub_clean_masks.py', path_list_masks[k+1]])
    
    # load cleaned masks
    masks = decodeMasks(readJson(path_file_masks))
    
    # init container
    to_df = []
    
    columns = ['image_name','SA_id','area_segm']
    columns += ['height', 'width', 'area_bbox']
    columns += ['cat','score']
    columns += [str(i) for i in range(768)]
    
    for mask in masks:
        # add columns to link with SA annotation
        row = [timestamp_id, mask['id'],mask['area']]
        
        # crop mask
        offset = 10
        cropped = cropMask(image,mask,offset)
        
        # save size
        h,w = cropped[:,:,0].shape
        row += [h-2*offset,w-2*offset,(h-2*offset)*(w-2*offset)]
        
        # save classification
        score, cat, _ = computeScore(cropped,prompt_input_categories,clp)
        row += [cat,score]
        
        #save embedding
        mask_embedding = getEmbedding(cropped,clp)
        row += [i for i in mask_embedding]
        
        # add to container
        to_df.append(row)
        
    # create df and save it 
    df = pd.DataFrame(to_df,columns=columns)
    writeDf(join(path_dir_output,getId(path_file_masks)+'.csv'), df)

'''path_dir_images_subsets_compress = join(path_root_image_subsets,channel,'compress')
os.makedirs(path_dir_images_subsets_compress,exist_ok=True)
random.seed(10)
for path_file_image in random.sample(path_list_classified_images,350):
    shutil.copy(path_file_image,join(path_dir_images_subsets_compress, getName(path_file_image)))'''