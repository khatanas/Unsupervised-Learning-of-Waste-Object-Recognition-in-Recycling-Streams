# for video_processing and image formating
max_width = 2064
min_light = 20
missed = 8
deadtime = 1

# embeddings threads
batch_size = 25
nb_threads = 4

# mask cleaning
th_min_area = 2000
th_iou_duplicates = .95

# embeddings
emb_dim = 768
nb_rows_max = 30*1000
crop_offset = 10

# GT 
nb_images_GT = 100

# faiss
ids_dim = 3
max_size_index = 50*1000

# faiss manual annotation
k_quick_candidates = 5

# faiss image retrieval
kNN_graph = 3
nb_min_image = 3
nb_max_image = 2500
nb_min_round = 1
nb_max_round = 25
kNN_egg = 3
k_out_of_kNN_egg = 3
nb_manual_annotations = 20
nb_augment_cat_index = 10

# CutLER
width_cutler = 1280

# UI window
display_width = 1000

# model architectures
arch_CLIP = 'ViT-L/14@336px'
arch_SA = "vit_h"
