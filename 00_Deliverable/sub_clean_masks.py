from helper.annotations import decodeMasks,cleanMasks,encodeMasks
from helper.tuning import readJson,writeJson
from helper.common_libraries import sys

path_file_masks = sys.argv[1]
idx = int(sys.argv[2])

masks = decodeMasks(readJson(path_file_masks))
new_masks = cleanMasks(masks)
writeJson(path_file_masks,encodeMasks(new_masks))

#print(f'cleaned file {idx+1}: {len(masks)} ==> {len(new_masks)}')