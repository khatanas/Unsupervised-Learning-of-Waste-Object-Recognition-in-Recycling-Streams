import sys
from os.path import join

# roots 
path_root = '/home/dkhatanassia/PDM/00_Deliverable'
path_root_libs = join(path_root,'libraries')
path_root_tests = join(path_root,'tests')
path_root_datasets = join(path_root,'datasets')
path_root_GT = join(path_root,'GT')


# for video processing
path_root_videos = '/home/dkhatanassia/drives/arc/cameras'
path_root_images_from_src = join(path_root,'00_images_from_src')
path_root_masks_from_images = join(path_root,'01_masks_from_images')
path_root_embeddings_from_masks = join(path_root,'02_embeddings_from_masks')
path_root_manual_annotation = join(path_root,'03_manual_annotation')
path_root_image_retrieval = join(path_root, '04_image_retrieval')


# libraries
path_lib_SA = join(path_root_libs, 'segment-anything')
sys.path.append(path_lib_SA)
path_file_SA_checkpoint = join(path_lib_SA, "sam_vit_h_4b8939.pth")

path_lib_CutLER = join(path_root_libs, 'CutLER')
sys.path.append(path_lib_CutLER)

path_lib_detectron2 = join(path_root_libs, 'detectron2')
sys.path.append(path_lib_detectron2)