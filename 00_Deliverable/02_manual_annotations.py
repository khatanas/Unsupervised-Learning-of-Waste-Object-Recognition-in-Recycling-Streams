from config.paths import *
from config.parameters import nb_images_GT, k_candidates

from helper.scriptIO import *
from helper.coco import *
from helper.faiss import *
from helper.dataframes import *
from helper.tuning import readJson,writeJson,sortedUnique
from helper.paths import getId,initPath,collectPaths
from helper.annotations import cleanMasks,encodeMasks,decodeMasks
from helper.common_libraries import subprocess,random,torch

from SA_annotationTool import annotateSA,printInstructions
os.environ["MKL_THREADING_LAYER"] = "GNU"
torch.cuda.empty_cache()
###################################### SHARED CONFIG  ##########################################
# channel to annotate
channel,path_dir_src = requestChannel(path_root_images_from_src)
initPath(join(path_root_GT,channel))

# path to json GT, where are stored the available categories
path_file_coco_annotations = join(path_root_GT,channel,f'{channel}_coco_annotations.json')

# get/create coco_annotation file
if not exists(path_file_coco_annotations):
    path_list_images = collectPaths(path_dir_src)
    path_list_images_GT = random.sample(path_list_images,nb_images_GT)
    initCocoAnnotations(path_file_coco_annotations,path_list_images_GT)

if requestBool('GT annotation'):####### GT annotation ###########################################
    #************************************* CONFIG ***********************************************
    random.seed(10)
    
    # path to output
    path_root_output = initPath(join(path_root_GT,channel))
    path_dir_output_anns = initPath(join(path_root_output,'ann_images'))
    
    #******************************** annotation process *****************************************
    while True:
        # add categories + select category
        coco_annotations = displayCategories(path_file_coco_annotations)
        nb_categories = len(coco_annotations['categories'])
        
        if requestBool("Launch annotation process"):
            
            # init annotation tool
            cat = requestInt("Category id", a=90001, b=int('9'+str(nb_categories).zfill(4)))
            
            # launch annotation tool
            printInstructions()
            torch.cuda.empty_cache()
            path_list_file_to_clean = annotateSA(path_dir_output_anns, getPathListFromCoco(coco_annotations), cat)
            
            # if some annotations have been added/removed
            if len(path_list_file_to_clean)>0:
                # clean new annotations
                for path in path_list_file_to_clean: writeJson(path,encodeMasks(cleanMasks(decodeMasks(readJson(path)))))
                
                # update coco_annotations file
                new_annotations = []
                for path in collectPaths(path_dir_output_anns):
                    anns = readJson(path)
                    for k,ann in enumerate(anns):
                        ann['image_id']=getCocoImageId(path, coco_annotations)
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
    first_run = True
    #******************************** annotation process *****************************************
    while True:
        if not first_run and requestBool('Terminate annotations process'):break
        else: first_run = False
        
        # add categories/select a category
        coco_annotations = displayCategories(path_file_coco_annotations)
        nb_categories = len(coco_annotations['categories'])
        cat = requestInt("Category id", a=90001, b=int('9'+str(nb_categories).zfill(4)))
        
        while True:
            # ==> get path_list_images to be displayed during annotation process
            # if some annotations exists for the selected category
            if (str(cat) in image_dict.keys()):
                # optionA: search {k_candidates} images with annotations similar to existing annotations
                if requestBool('Retrieve image candidates'):                
                    if not index_initialized:
                        # request path to embedding database and build faiss index
                        points,path_dir_src_xb = requestPtsPerSide(path_root_embeddings_from_masks)
                        print('\nBuilding database')
                        index,xq,ids_xb = initQuickInstanceRetrieval(join(path_dir_src_xb,channel,'df_merged'),path_dir_output_merged,cat)
                        index_initialized = True
                    else: _,xq,_= initQuery(path_dir_output_merged,cat)
                    # get new image by performing a query on the index
                    _,I = index.search(xq,k_candidates)
                    path_list_images = [getImagePath(item) for item in newTimestamps(ids_xb,I)]
                # optionB: collect images where annotations for the selected category exists
                elif requestBool("Browse already annotated"): path_list_images = list(set([getImagePath(item) for item in image_dict[str(cat)]['0']]))
            # no annotations exists, collect all available images. 
            else: path_list_images = collectPaths(path_dir_src)
            random.shuffle(path_list_images)
            
            # launch annotation tool
            printInstructions()
            torch.cuda.empty_cache()
            path_list_file_to_update = annotateSA(path_dir_output_anns, path_list_images, cat)
            
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
                subprocess.run(['python', 'sub_embeddings_multithreading.py',path_dir_output_anns, path_dir_output_emb, str(4)])
                
                print('Merging dataframes')
                df_merged = mergeDf(path_dir_output_emb,path_dir_output_merged)
                
                # image dictionnary
                for cat in sortedUnique(extractValues(df_merged,'category_id')):
                    df_cat = filterCat(df_merged,cat)
                    print(f'{cat} - {df_cat.shape[0]} annotations available')
                    image_cat = image_dict[str(cat)] if (str(cat) in image_dict.keys()) else dict()
                    image_cat['0'] = list(set([item for item in extractValues(df_cat,'timestamp_id')]))
                    image_dict[str(cat)] = image_cat 
                writeJson(path_file_image_dict,image_dict)
            if requestBool('Change annotation category'):break