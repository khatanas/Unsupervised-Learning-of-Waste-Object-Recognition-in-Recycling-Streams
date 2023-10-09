from config_.paths import *
from config_.parameters import nb_images_GT, k_quick_candidates, nb_threads

from helper.scriptIO import *
from helper.coco import *
from helper.faiss import *
from helper.dataframes import *
from helper.tuning import readJson,writeJson,sortedUnique
from helper.paths import getId,initPath,collectPaths
from helper.annotations import cleanMasks,encodeMasks,decodeMasks
from helper.SA_annotationTool import annotateSA,printInstructions
from helper.common_libraries import subprocess,random,torch

os.environ["MKL_THREADING_LAYER"] = "GNU"
torch.cuda.empty_cache()
###################################### SHARED CONFIG  ##########################################
# channel to annotate
channel,path_dir_src = requestChannel(path_root_images_from_src)

# create/load coco_annotation file is necessary
path_root_output = initPath(join(path_root_GT,channel))
path_file_coco_annotations = join(path_root_GT,channel,f'{channel}_coco_annotations.json')
coco_annotations = readJson(path_file_coco_annotations) if exists(path_file_coco_annotations) else initCocoAnnotations(path_file_coco_annotations)
exit_requested = False
if requestBool('GT annotation'):####### GT annotation ###########################################
    #************************************* CONFIG ***********************************************
    random.seed(10)
    
    # path to output
    path_dir_output_anns = initPath(join(path_root_output,'ann_images'))
    
    #******************************** annotation process *****************************************
    while True:
        # add categories + select category
        coco_annotations = augmentCategories(path_file_coco_annotations)
        nb_categories = len(coco_annotations['categories'])
        
        if requestBool("Launch annotation process"):
            
            # init annotation tool
            categories = [item['id'] for item in coco_annotations['categories']]
            cat = requestInt("Category id", a=min(categories), b=max(categories))
            
            # get images
            path_list_images = collectPaths(join(path_root_images_from_src,channel))
            random.shuffle(path_list_images)
            path_list_images = getPathListFromCoco(coco_annotations) + path_list_images[:nb_images_GT]
            
            # launch annotation tool
            printInstructions()
            torch.cuda.empty_cache()
            path_list_file_to_update = annotateSA(path_dir_output_anns, path_list_images, cat)
            
            # if some annotations have been added/removed
            if len(path_list_file_to_update)>0:
                # clean new annotations
                for path_file_masks in path_list_file_to_update: 
                    writeJson(path_file_masks,encodeMasks(cleanMasks(decodeMasks(readJson(path_file_masks)))))
                    # delete empty annotation files and associated GT image field
                    if len(readJson(path_file_masks))==0:
                        os.remove(path_file_masks)
                        del coco_annotations['images'][getCocoName(path_file_masks)] 
                    # add image to GT
                    else: coco_annotations['images'].append(initCocoImage(getImagePath(getId(path_file_masks)),len(coco_annotations['images'])))
                
                # update coco_annotations file
                new_annotations = []
                for path_file_masks in collectPaths(path_dir_output_anns):
                    anns = readJson(path_file_masks)
                    for k,ann in enumerate(anns):
                        ann['image_id']=getCocoImageId(path_file_masks, coco_annotations)
                        new_annotations.append(ann)
                coco_annotations['annotations'] = new_annotations
                coco_annotations = makeIdUnique(path_file_coco_annotations,coco_annotations)
                
                if requestBool('Terminate annotations process'):break
        else: break
        
else:########################## Semi-supervised annotation ##########################################
    #*********************************** CONFIG *****************************************************
    # path to output
    path_root_output = join(path_root_manual_annotation,channel)
    path_dir_output_anns = initPath(join(path_root_output,'ann_images'))
    path_dir_output_emb = initPath(join(path_root_output,'df_images'))
    path_dir_output_merged = initPath(join(path_root_output,'df_merged'))
    path_file_image_dict = join(path_root_output,'image_dict.json')
    image_dict = readJson(path_file_image_dict) if exists(path_file_image_dict) else dict()
    
    index_initialized = False
    exit_requested = False
    #******************************** annotation process *****************************************
    while True:
        if exit_requested: break
        
        # add categories/select a category
        coco_annotations = augmentCategories(path_file_coco_annotations)
        nb_categories = len(coco_annotations['categories'])
        categories = [item['id'] for item in coco_annotations['categories']]
        cat = requestInt("Category id", a=min(categories), b=max(categories))
        while True:
            #******************** get path_list_images to be displayed during annotation process ************
            # CASE A: annotations exist ==> Get new image using FAISS and the already annotated images
            if (str(cat) in image_dict.keys()) and requestBool('Fetch new images'):          
                if not index_initialized:
                    # request path to embedding database and build faiss index
                    points,path_dir_src_xb = requestPtsPerSide(path_root_embeddings_from_masks)
                    print('\nBuilding database')
                    index,xq,ids_xb = initQuickInstanceRetrieval(join(path_dir_src_xb,channel,'df_merged'),collectPaths(path_dir_output_merged)[0],cat)
                    index_initialized = True
                    
                else: _,xq,_= initQuery(collectPaths(path_dir_output_merged)[0],cat)
                # get new image by performing a query on the index
                _,I = index.search(xq,k_quick_candidates)
                path_list_images = [getImagePath(item) for item in newTimestamps(ids_xb,I)]
                
            # CASE B: annotations exist ==> display annotated images
            elif (str(cat) in image_dict.keys()) and requestBool("Browse already annotated"):
                path_list_images = list(set([getImagePath(item) for item in image_dict[str(cat)]['manual']]))
            
            else: 
                # CASE C: annotation doesn't exist OR caseA,caseB rejected ==> provide timestamp of know image
                if requestBool('Known timestamp'):
                    _,path_file_image = requestTimestamp(channel)
                    path_list_images = [path_file_image]
                
                # CASE D: annotation doesn't exist OR caseA,caseB,caseC rejected ==> load all available image, shuffle them
                else:
                    path_list_images = collectPaths(path_dir_src)
                    random.shuffle(path_list_images)
            #**************************************************************************************************
            
            # launch annotation tool
            printInstructions()
            torch.cuda.empty_cache()
            path_list_file_to_update = annotateSA(path_dir_output_anns, path_list_images, cat)
            
            #**************************** Collect embeddings and update image_dict ***************************
            if len(path_list_file_to_update)>=0:
                # ==> compute new embeddings
                print('\nEmbeddings collection')        
                # file cleaning
                for path in path_list_file_to_update:
                    # delete empty annotation files
                    if len(readJson(path))==0: os.remove(path)
                    # delete embeddings being re-computed
                    path_file_embeddings = join(path_dir_output_emb, getId(path)+'.csv')
                    if exists(path_file_embeddings):os.remove(path_file_embeddings)
                # launch computation
                torch.cuda.empty_cache()
                subprocess.run(['python', 'sub_embeddings_multithreading.py',path_dir_output_anns, path_dir_output_emb, str(nb_threads)])
                
                print('Merging dataframes')
                _,df_merged = mergeDf(path_dir_output_emb,path_dir_output_merged)
                
                # image dictionnary
                for item in sortedUnique(extractValues(df_merged,'category_id')):
                    df_cat = filterCat(df_merged,item)
                    print(f'{item} - {df_cat.shape[0]} annotations available')
                    image_cat = image_dict[str(item)] if (str(item) in image_dict.keys()) else dict()
                    image_cat['manual'] = list(set([item for item in extractValues(df_cat,'timestamp_id')]))
                    image_dict[str(item)] = image_cat 
                writeJson(path_file_image_dict,image_dict)
            #**************************************************************************************************
            if requestBool('Terminate annotations process'): 
                exit_requested = True
                break
            elif requestBool('Change annotation category'):break