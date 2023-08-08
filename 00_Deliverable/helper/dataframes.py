from config.parameters import emb_dim,crop_offset,nb_rows_max

from helper.paths import getImagePath,collectPaths
from helper.tuning import imreadRGB,sortedUnique,writeJson
from helper.annotations import cropMask
from helper.CLIP import getEmbedding
from helper.common_libraries import pd,join,np


filterCat = lambda df,cat: df[df['category_id']==cat]
filterAoI = lambda df,AoI: df[df['AoI']==AoI]
filterHead = lambda df,head: df[df['is_head']==head]
filterTimestamp = lambda df, list_timestamp: df[~df['timestamp_id'].isin(list_timestamp)]
extractValues = lambda df,column: [int(item) if type(item)!=str else item for item in list(df[column].values)]
extractIds = lambda df: [extractValues(df,'timestamp_id'),extractValues(df,'SA_id')]
extractEmbeddings = lambda df: np.array(df.iloc[:,-emb_dim:],dtype='float32',order='C')
stackDfs = lambda list_df: pd.concat(list_df, axis=0)

def writeDf(path_file_csv,df):
    df.to_csv(path_file_csv, index=False)


def readDf(path_file_csv):
    return pd.read_csv(path_file_csv)


def mergeDf(path_dir_src,path_dir_output,nb_rows_max=nb_rows_max):
    acc = 0
    count = 0
    merged_dict = {}
    
    dfs = []
    for path in collectPaths(path_dir_src):
        df = readDf(path)
        dfs.append(df)
        timestamps_ids = sortedUnique(extractValues(df,'timestamp_id'))
        for unique in timestamps_ids: merged_dict[unique]=count
        acc += dfs[-1].shape[0]
        if acc>=nb_rows_max:
            df = stackDfs(dfs)
            writeDf(join(path_dir_output,f'merged{count}.csv'),df)
            print(f'saved merged{count}.csv')
            count += 1
            acc = 0
            dfs = []
    df = stackDfs(dfs)
    timestamps_ids = sortedUnique(extractValues(df,'timestamp_id'))
    for unique in timestamps_ids: merged_dict[unique]=count
    writeDf(join(path_dir_output,f'merged{count}.csv'),df)
    print(f'saved merged{count}.csv')
    
    writeJson(path_dir_output.replace('df_merged','merged_dict.json'),merged_dict)
    return df


def collectEmbeddings(masks, path_dir_output, clp):
    path_file_image = getImagePath(masks[0]['timestamp_id'])
    image = imreadRGB(path_file_image)
    
    # init container
    to_df = []
    
    # init columns
    columns = ['timestamp_id','SA_id']
    columns += ['is_head', 'solo', 'head_id']
    columns += ['area_segm', 'height', 'width', 'area_bbox']
    columns += ['AoI', 'category_id']#,'IR_source','IR_target']
    columns += ['imread']+[str(i) for i in range(768)]
    
    for mask in masks:
        # id info: ['timestamp_id','SA_id']
        row = [mask['timestamp_id'], mask['id']]
        
        # overlapping info
        row += [mask['is_head'],mask['solo'],mask['head_id']]
        
        # crop mask
        offset = crop_offset
        cropped = cropMask(image,mask,offset)
        
        # area info: ['area_segm','height', 'width', 'area_bbox']
        h,w = cropped.shape[:2]
        h -= 2*offset
        w -= 2*offset
        row += [mask['area'],h,w,h*w]
        
        # classification info: ['AoI','category_id']
        row += [mask['AoI'], mask['category_id'] if 'category_id' in mask.keys() else -1]
        
        # embedding info: [imread, str(i) for i in range(768)]
        mask_embedding = getEmbedding(cropped,clp)
        row += ['Gray' if len(cropped.shape)==2 else 'RGB'] + [i for i in mask_embedding]
        
        # add to container
        to_df.append(row)
        
    # create df and save it 
    df = pd.DataFrame(to_df,columns=columns)
    writeDf(join(path_dir_output,masks[0]['timestamp_id']+'.csv'), df)