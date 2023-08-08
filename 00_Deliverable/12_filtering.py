from helper._config_paths import *
from helper._config_classification import *
from helper._scriptIO import *
from helper._CLIP import *
from helper._tuning import readJson,imreadRGB
from helper._annotations import decodeMasks, cropMask
from helper._paths import getImagePath,getId,getName,initPath
from helper._df import *

import random
############################################## CONFIG #######################################################################
points_per_side, path_dir_src = requestPtsPerSide(path_root_embeddings)
channel, path_dir_src = requestChannel(path_dir_src)
path_list_dfs = collectPaths(join(path_dir_src,'df_images'))

taxononmy, prompt_input_categories, names_all, names_subparts = getTaxonomy(channel)

# output
path_dir_output = join(path_dir_src,'filtered')

