import pandas as pd

def buildDf(path_list):
    # load csv containing outputs from clip
    df = pd.read_csv(path_list[0])
    for i in path_list[1:]:
        tmp = pd.read_csv(i)
        df = pd.concat([df, tmp], axis=0)
    return df