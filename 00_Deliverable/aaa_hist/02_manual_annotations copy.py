from config.paths import *
from config.parameters import nb_images_GT,nb_max_rows

from helper.scriptIO import *
from helper.coco import *
from helper.tuning import readJson,writeJson
from helper.paths import getId,initPath,collectPaths
from helper.dataframes import buildLimitedDf,filterCat,extractValues
from helper.annotations import cleanMasks,encodeMasks,decodeMasks
from helper.common_libraries import subprocess,random,torch

from SA_annotationTool import annotateSA,printInstructions
os.environ["MKL_THREADING_LAYER"] = "GNU"

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

if requestBool('GT annotation'):####### GT annotation ###############################
    random.seed(10)
    
    # path to output
    path_root_output = initPath(join(path_root_GT,channel))
    
    # annotation process
    while True:
        # display categories and permits to augment the list
        coco_annotations = displayCategories(path_file_coco_annotations)
        nb_categories = len(coco_annotations['categories'])
        
        # annotate
        if requestBool("Launch annotation process"):
            
            # init annotation tool
            printInstructions()
            cat = requestInt("Category id", a=90001, b=int('9'+str(nb_categories).zfill(4)))
            path_list_file_to_clean = []
            path_dir_output_anns = initPath(join(path_root_output,'ann_images'))
            path_list_images_GT = getPathListFromCoco(coco_annotations)
            
            # launch annotation tool
            path_list_file_to_clean += annotateSA(path_dir_output_anns, path_list_images_GT, cat)
            
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
        
else:###################### Semi-supervised annotation ##########################################
    # path to output
    path_root_output = join(path_root_manual_annotation,channel)
    path_file_image_dict = join(path_root_output,'image_dict.json')
    
    path_list_file_to_update = []
    while True:
        # add categories
        coco_annotations = displayCategories(path_file_coco_annotations)
        nb_categories = len(coco_annotations['categories'])
        
        # init annotation tool
        printInstructions()
        cat = requestInt("Category id", a=90001, b=int('9'+str(nb_categories).zfill(4)))
        path_dir_output_anns = initPath(join(path_root_output,'ann_images'))
        path_list_images = collectPaths(path_dir_src)
        random.shuffle(path_list_images)
        
        # launch annotation tool
        path_list_file_to_update += annotateSA(path_dir_output_anns, path_list_images, cat)
        if requestBool('Terminate annotations process'):break
        
    if len(path_list_file_to_update)>=0:
        # compute new embeddings
        print('Embeddings collection')
        # init embeddings directory
        path_dir_output_emb = initPath(join(path_root_output,'df_images'))
        path_dir_output_merged = initPath(join(path_root_output,'df_merged'))
        
        # file cleaning
        for path in path_list_file_to_update:
            # delete empty annotation files
            if len(readJson(path))==0: os.remove(path)
            # delete embeddings being re-computed
            path_file_embeddings = join(path_dir_output_emb, getId(path)+'.csv')
            if exists(path_file_embeddings):os.remove(path_file_embeddings)
            
        torch.cuda.empty_cache()
        subprocess.run(['python', 'sub_embeddings_multithreading.py',path_dir_output_anns ,path_dir_output_emb, str(3)])
        
        print('Merging dataframes')
        df_merged = buildLimitedDf(collectPaths(path_dir_output_emb),nb_max_rows,path_dir_output_merged)
        
        # image dictionnary
        path_file_image_dict = initPath(join(path_root_output,'image_dict.json'))
        image_dict = {}  if not exists(path_file_image_dict) else readJson(path_file_image_dict)
        for cat in df_merged['category_id'].unique():
            df_cat = filterCat(df_merged,cat)
            image_dict[cat][0]=[timestamp_id for timestamp_id in extractValues(df_cat,'timestamp_id')]
        writeJson(path_file_image_dict,image_dict)



'''        # create dfs where embeddings are classified per category
        print('Classify embeddings...')
        for cat in [category['id'] for category in coco_annotations['categories']]:
            # created category-df from image-df
            path_dir_embeddings_cat = initPath(join(path_root_output,'df_categories'))
            path_file_embeddings_cat = join(path_dir_embeddings_cat,str(cat)+'.csv')
            df_cat = collectCategory(collectPaths(path_dir_output_emb),cat)
            writeDf(path_file_embeddings_cat,df_cat)
            
            # add images to image_dict
            if not exists(path_file_image_dict):
                image_dict = {cat:[]}
                writeJson(path_file_image_dict,image_dict)
            image_dict = readJson(path_file_image_dict)
            
            image_dict[str(cat)]=list(df_cat['timestamp_id'].unique())
            writeJson(path_file_image_dict,image_dict)'''
