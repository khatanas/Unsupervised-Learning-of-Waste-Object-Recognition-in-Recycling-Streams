from helper.annotations import decodeMasks,cleanMasks,encodeMasks
from helper.tuning import readJson,writeJson
from helper.common_libraries import sys

# path to masks to clean
path_file_masks = sys.argv[1]

# load and decode
masks = decodeMasks(readJson(path_file_masks))
# clean
new_masks = cleanMasks(masks,th_area=2500)
assert (all([k_th == mask['id'] for k_th,mask in enumerate(new_masks)]))

# encode and write
writeJson(path_file_masks,encodeMasks(new_masks))