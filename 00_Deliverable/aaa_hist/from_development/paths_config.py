from os.path import join

# roots 
path_root = '/home/dkhatanassia/PDM/01_Development'
path_root_vp = join(path_root,'00_video_processing')


# for video processing
path_root_videos = '/home/dkhatanassia/drives/arc/cameras'
path_root_images_from_videos = join(path_root_vp,'00_images_from_videos')
path_root_masks_from_images = join(path_root_vp,'01_masks_from_images')
path_dir_lists_of_subsets = join(path_root_vp,'02_lists_of_subsets')






root = '/home/dkhatanassia/PDM'

# for video processing
path_dir_videos = '/home/dkhatanassia/drives/arc/cameras'
path_dir_img_from_videos = join(root,'img_from_videos')
path_dir_masks_from_imgs = join(root,'masks_from_img')
path_dir_clean_SA_masks = join(path_dir_masks_from_imgs, '01_SA_masks_cleaned')

# available datasets
path_dir_jerry_sp10 = '/home/dkhatanassia/PDM/01_datasets/jerry_sp10'
path_dir_jerry_videos = '/home/dkhatanassia/PDM/01_datasets/jerry_videos'