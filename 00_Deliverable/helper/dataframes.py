from config_.parameters import emb_dim,crop_offset,nb_rows_max

from helper.paths import getImagePath,collectPaths
from helper.tuning import imreadRGB,sortedUnique,writeJson
from helper.annotations import cropMask
from helper.CLIP import getEmbedding,computeScore
from helper.common_libraries import pd,join,np

#**************************************************************************************************************
# a few manipiulation lambda functions
filterCat = lambda df,cat: df[df['category_id']==cat]
filterAoI = lambda df,AoI: df[df['AoI']==AoI]
filterHead = lambda df,head: df[df['is_head']==head]
filterSolo = lambda df,solo: df[df['solo']==solo]
cutTimestamp = lambda df, list_timestamp: df[~df['timestamp_id'].isin(list_timestamp)]
cutIds = lambda df, ids: df[np.logical_and(~df['timestamp_id'].isin(ids[0]), ~df['SA_id'].isin(ids[1]))]
extractValues = lambda df,column: [int(item) if type(item)!=str else item for item in list(df[column].values)]
extractIds = lambda df: [extractValues(df,'timestamp_id'),extractValues(df,'SA_id')]
extractEmbeddings = lambda df: np.array(df.iloc[:,-emb_dim:],dtype='float32',order='C')
stackDfs = lambda list_df: pd.concat(list_df, axis=0)


def writeDf(path_file_csv,df):
    """
    Saves df as csv file. Could potentially be changed
    """
    df.to_csv(path_file_csv, index=False)


def readDf(path_file_csv):
    """
    Read df from csv file. Could potentially be changed
    """ 
    return pd.read_csv(path_file_csv)


def mergeDf(path_dir_src,path_dir_output,nb_rows_max=nb_rows_max):
    """
    Merges the dfs located in {path_dir_src}, up to {nb_rows_max} per file, and stores them at {path_dir_output} 
    """
    # to keep track of #total masks, #AoI masks, #Head masks, within AoI
    stats = [0,0,0]
    
    # to keep to of which image is stored in which merged file
    merged_dict = {}
    
    # to keep track of #rows being stacked together
    acc = 0
    count = 0
    dfs = []
    for path in collectPaths(path_dir_src):
        # read and store
        df = readDf(path)
        dfs.append(df)
        # match timestamp with merged_file_id
        timestamps_ids = sortedUnique(extractValues(df,'timestamp_id'))
        for unique in timestamps_ids: merged_dict[unique]=count
        # count #rows and if acc>max_nb_rows, store file and update stats
        acc += df.shape[0]
        if acc>=nb_rows_max:
            df = stackDfs(dfs)
            writeDf(join(path_dir_output,f'merged{count}.csv'),df)
            print(f'saved merged{count}.csv')
            
            stats[0] += acc
            stats[1] += filterAoI(df,1).shape[0]
            stats[2] += filterHead(filterAoI(df,1),1).shape[0]
            
            acc = 0
            count += 1
            dfs = []
    # save final
    df = stackDfs(dfs)
    timestamps_ids = sortedUnique(extractValues(df,'timestamp_id'))
    for unique in timestamps_ids: merged_dict[unique]=count
    writeDf(join(path_dir_output,f'merged{count}.csv'),df)
    print(f'saved merged{count}.csv')
    
    stats[0] += acc
    stats[1] += filterAoI(df,1).shape[0]
    stats[2] += filterHead(filterAoI(df,1),1).shape[0]
    writeJson(path_dir_output.replace('df_merged','merged_dict.json'),merged_dict)
    return stats,df


def collectEmbeddings(masks, path_dir_output, clp, encoded_text_prompt=[]):
    """
    Computes the image embedding of each mask in masks if no {encoded_text_prompt} are provided.
    Otherwise, doesn't compute the image embedding but the CLIP classification score and category.
    {encoded_text_prompt} input must be: [tokenized_{encoded_text_prompt}, category_ids].
    The computed and extracted information is stored in a df, saved at {path_dir_output}
    """
    path_file_image = getImagePath(masks[0]['timestamp_id'])
    image = imreadRGB(path_file_image)
    
    # init container
    to_df = []
    
    # init columns
    columns =  ['timestamp_id','SA_id']
    columns += ['is_head', 'solo', 'head_id','AoI']
    columns += ['area_segm', 'height', 'width', 'area_bbox']
    columns += ['in_GT','GT_id','GT_category']
    columns += ['imread']+(['category_id'] if not encoded_text_prompt else ['score','category_id'])+[str(i) for i in range(768)]
    
    for mask in masks:
        # id info: ['timestamp_id','SA_id']
        row = [mask['timestamp_id'], mask['id']]
        
        # spatial info
        row += [mask['is_head'],mask['solo'],mask['head_id'],mask['AoI']]
        
        # crop mask
        offset = crop_offset
        cropped = cropMask(image,mask,crop_offset)
        
        # area info: ['area_segm','height', 'width', 'area_bbox']
        h,w = cropped.shape[:2]
        h -= 2*offset
        w -= 2*offset
        row += [mask['area'],h,w,h*w]
        
        # GT info: ['in_GT',GT_id,'category_GT']
        for name,default in zip(['in_GT','GT_id','GT_category'],[0,-1,-1]):
            row += [mask[name] if name in mask.keys() else default]
        
        # embedding info: [imread]
        row += ['Gray' if len(cropped.shape)==2 else 'RGB']
        
        if encoded_text_prompt:
            #==> CLIP text prompt, for test only. The embedding is not computed. 
            # [score, category_id, embedding]
            score, cat_idx,_ = computeScore(cropped,encoded_text_prompt[0],clp)
            category_id = encoded_text_prompt[1][cat_idx]
            row += [score,category_id]+[-1 for _ in range(emb_dim)]
        else:
            # [category_id, embedding]
            mask_embedding = getEmbedding(cropped,clp)
            row += [mask['category_id'] if 'category_id' in mask.keys() else -1] + [i for i in mask_embedding]
        
        # add to container
        to_df.append(row)
        
    # create df and save it 
    df = pd.DataFrame(to_df,columns=columns)
    writeDf(join(path_dir_output,masks[0]['timestamp_id']+'.csv'), df)