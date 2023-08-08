from config.paths import *
from helper.scriptIO import *
from helper.SA_annotationTool import *
from helper.coco import initDict
from helper.paths import getName,getImagePath,getId,initPath


import shutil
import random
random.seed(10)


print('When the image is displayed:\n',
      'i) Click to add some points (multiple clicks possible):\n',
      '     - left click: add point to include (+)\n',
      '     - right click: add point to avoid (-)\n',
      '     ==> when done with clicking, press any key\n\n',
      
      'ii) The diplayed image shows clicks - available actions:\n',
      '     - press "r" to remove last click\n',
      '     - press "x" to remove all clicks\n',
      '     - press "v" to validate current annotation\n',
      '     - press "p" to pop current annotation from json file\n\n',
      
      'iii) The diplayed image do not show any click - available actions:\n',
      '     - press "c" to cancel previous annotation\n',
      '     - press "d" to save current annotations to json file and move to next image\n',
      '     - press "q" to quit annotation process\n'
)

# masks locations
points_per_side, path_dir_src = requestPtsPerSide(path_root_masks_from_images)
channel, path_dir_src = requestChannel(path_dir_src)
path_list_masks = collectPaths(path_dir_src)

# random subset and corresponding image location
path_list_masks_k = random.sample(path_list_masks,100)
path_list_images_k = [getImagePath(getId(path)) for path in path_list_masks_k]

# output location 
path_root_output = join(path_root_mask_classification,points_per_side,channel)
path_dir_images_k = join(path_root_output,'images')

if not exists(path_dir_images_k):
      _ = initPath(path_dir_images_k)
      print('Copying images...')
      for path_file_image in path_list_images_k:
            shutil.copy(path_file_image, join(path_dir_images_k,getName(path_file_image)))

# init GT file path and taxonomy
path_file_annotations = join(path_root_output,f'{channel}_annotations.json')
cat = requestCategory(channel)

if not exists(path_file_annotations): _ = initDict(path_dir_images_k, path_file_json=path_file_annotations,taxonomy_channel=channel)
_ = annotateSA(path_file_annotations, path_dir_images_k, SamPredictor(initSam()), cat=cat)

