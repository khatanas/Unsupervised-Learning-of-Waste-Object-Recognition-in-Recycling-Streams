from config.paths import *
from config.parameters import nb_images_GT

from helper.scriptIO import *
from helper.coco import *
from helper.tuning import readJson,writeJson
from helper.paths import getId,initPath,collectPaths
from helper.dataframes import collectCategory,writeDf
from helper.annotations import cleanMasks,encodeMasks,decodeMasks
from helper.common_libraries import subprocess,random,torch

from SA_annotationTool import annotateSA

os.environ["MKL_THREADING_LAYER"] = "GNU"

print('When the image is displayed:\n',
      'i) Click to add some points (multiple clicks possible):\n',
      '     - left click: add point to include (+)\n',
      '     - right click: add point to avoid (-)\n',
      '     ==> when done with clicking, press any key\n\n',
      
      'ii) The diplayed image shows clicks - available actions:\n',
      '     - press "w" to remove last click\n',
      '     - press "q" to remove all clicks\n',
      '     - press "s" to validate current annotation\n',
      '     - press "f" to pop current annotation from json file\n\n',
      
      'iii) The diplayed image do not show any click - available actions:\n',
      '     - press "c" to cancel previous annotation\n',
      '     - press "d" to save current annotations to json file and move to next image\n',
      '     - press "a" to save current annotations to json file and move to previous image\n',
      '     - press "q" to save current annotations and quit annotation process'
)
print('')

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
                  cat = requestInt("Category id", a=90001, b=int('9'+str(nb_categories).zfill(4)))
                  to_be_cleaned = []
                  path_dir_anns = initPath(join(path_root_output,'ann_images'))
                  path_list_images_GT = getPathListFromCoco(coco_annotations)
                  
                  # launch annotation tool
                  to_be_cleaned += annotateSA(path_dir_anns, path_list_images_GT, cat)
                  
                  if to_be_cleaned>0:
                        # clean new annotations
                        for path in to_be_cleaned: writeJson(path,encodeMasks(cleanMasks(decodeMasks(readJson(path)))))
                        
                        # update coco_annotations file
                        new_annotations = []
                        for path in collectPaths(path_dir_anns):
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
      
      to_be_updated = []
      while True:
            # add categories
            coco_annotations = displayCategories(path_file_coco_annotations)
            nb_categories = len(coco_annotations['categories'])
            
            # init annotation tool
            cat = requestInt("Category id", a=90001, b=int('9'+str(nb_categories).zfill(4)))
            path_dir_anns = initPath(join(path_root_output,'ann_images'))
            path_list_images = collectPaths(path_dir_src)
            random.shuffle(path_list_images)
            
            # launch annotation tool
            to_be_updated += annotateSA(path_dir_anns, path_list_images, cat)
            if requestBool('Terminate annotations process'):break
            
      torch.cuda.empty_cache()
      if len(to_be_updated)>0 :
            # compute embeddings
            path_dir_emb = initPath(join(path_root_output,'df_images'))
            subprocess.run(['python', 'sub_launch_threads_embeddings.py',path_dir_anns,path_dir_emb, str(3)])
            
            # delete empty files if any
            for path in to_be_updated:
                  if len(readJson(path))==0:
                        os.remove(path)
                        path_file_emb = join(path_dir_emb, getId(path)+'.csv')
                        if exists(path_file_emb):os.remove(path_file_emb) 
                        
            # created category-df from image-df
            path_dir_embeddings_cat = initPath(join(path_root_output,'df_categories'))
            for path in to_be_updated:
                  path_file_emb = join(path_dir_emb, getId(path)+'.csv')
                  df = readJson
            path_file_embeddings_cat = join(path_dir_embeddings_cat,str(cat)+'.csv')
            df_cat = collectCategory(collectPaths(path_dir_emb),cat)
            writeDf(path_file_embeddings_cat,df_cat)
            
            # add images to image_dict
            if not exists(path_file_image_dict):
                  image_dict = {cat:[]}
                  writeJson(path_file_image_dict,image_dict)
            image_dict = readJson(path_file_image_dict)
            
            image_dict[str(cat)]=list(df_cat['timestamp_id'].unique())
            writeJson(path_file_image_dict,image_dict)
