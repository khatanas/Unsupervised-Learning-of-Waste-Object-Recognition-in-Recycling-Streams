# for video_processing and image formating
max_width = 2064
min_light = 20
missed = 8
deadtime = 1

# embeddings threads
batch_size = 25
nb_threads = 3

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
k_candidates = 10
ids_dim = 3

# UI window
display_width = 1000

# model architectures
arch_CLIP = 'ViT-L/14@336px'
arch_SA = "vit_h"
