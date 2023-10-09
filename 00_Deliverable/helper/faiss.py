from config_.paths import *
from config_.parameters import *

from helper.dataframes import *
from helper.paths import collectPaths,initPath,getName
from helper.annotations import decodeMasks,filterMasks
from helper.tuning import readJson,sortedUnique, flattenList
from helper.common_libraries import join,np,torch,random,exists

import faiss

#**************************************************************************************************************
randomSample = lambda group,k: group.sample(k)
newTimestamps = lambda ids_xb,I: sortedUnique([ids_xb[0][k] for k in list(I.flatten())])


def initIndex(gpu):
    """
    Initializes an index as well as a container for both the embeddings and their ids
    """
    index = faiss.IndexFlatL2(emb_dim)
    if (torch.cuda.is_available() and gpu): index = faiss.index_cpu_to_gpu(faiss.StandardGpuResources(),0,index)
    xb = []
    xb_ids = [[] for _ in range(ids_dim)]
    return index, xb, xb_ids


def augmentIndex(index,df_xb,xb,ids_xb):
    """
    Adds the embeddings contained in {df_xb} to the index.
    Adds the embeddings contained in {df_xb} to the provided {xb}.
    Adds the embeddings ids to the provided {ids_xb}
    Returns the augmented index, xb, and ids_xb.
    """
    # adds the embeddings to the index
    index.add(extractEmbeddings(df_xb))
    
    #collect embeddings
    xb.append(extractEmbeddings(df_xb))
    
    # collect the info for the mapping
    ids_xb[0]+=extractValues(df_xb,'timestamp_id')
    ids_xb[1]+=extractValues(df_xb,'SA_id')
    ids_xb[2]+=extractValues(df_xb,'category_id')
    
    return index,xb,ids_xb


def initQuery(path_file_merged_xq, category_id=False, k=nb_manual_annotations):
    """
    Fetch the manually annotated dataframe, extracts the embeddings and its associated ids
    Returns all three elements
    """
    # get df
    df_xq = readDf(path_file_merged_xq)
    
    # specific category or subset of k instances of each category
    if category_id or k:
        df_filtered = []
        for cat in [category_id] if category_id else sortedUnique(extractValues(df_xq,'category_id')):
            df_cat = filterCat(df_xq,cat)
            # pick subset of annotations if requested
            df_cat = df_cat.sample(k) if k and df_cat.shape[0]>k else df_cat
            df_filtered.append(df_cat)
        df_xq = stackDfs(df_filtered)
    # extract embeddings, timestamp_id, SA_id from df
    xq = extractEmbeddings(df_xq)
    ids_xq = extractIds(df_xq)
    ids_xq.append(extractValues(df_xq,'category_id'))
    return df_xq, xq, ids_xq


def initQuickInstanceRetrieval(path_dir_merged_xb, path_file_merged_xq, category_id, gpu=False):
    """
    Builds an index with a subpart of the available data and reads the query data.
    Returns the index, the the query embeddings and the ids of the embeddings used to build the index
    """
    # extract embeddings from manual annotation
    _,xq, ids_xq = initQuery(collectPaths(path_file_merged_xq)[0],category_id)
    
    # init index and  iterate over subset of merged files
    index,xb, ids_xb = initIndex(gpu)
    
    # collect paths and pick {nb_file} of them at random (speed-up)
    nb_file = int(max_size_index/nb_rows_max)+1
    path_list_merged_xb = collectPaths(path_dir_merged_xb)
    random.shuffle(path_list_merged_xb)
    path_list_merged_xb = path_list_merged_xb[:nb_file]
    
    for path in path_list_merged_xb:
        # get df, remove seen image
        df_xb = readDf(path)
        df_xb = cutTimestamp(df_xb,ids_xq[0])
        
        # apply further filters
        df_xb = filterAoI(df_xb,1)
        df_xb = filterHead(df_xb,1)
        
        # augment index
        index,xb,ids_xb = augmentIndex(index,df_xb,xb,ids_xb)
        
    # ==> returns the index, the queries embeddings, the index mapping
    return index, xq, ids_xb


def initImageRetrieval(path_list_merged_xb, path_file_merged_xq, category_id, forbidden_timestamps, solo=False,head=True,AoI=True,gpu=True):
    """
    Builds an index to perform image retrieval using kNN graph.
    Returns the index, all embeddings used to perform queries, their associated ids, and the nb of manually annotated annotations available
    X stacks the embeddings as follows: 
    X = [all SAM embeddings without assigned categories (those are used to build the index) | manually annotated embeddings]
    Thus:
        - X[:index.ntotal] = all SAM embeddings without assigned categories 
        - X[-nb_annotations:] = manually annotated embeddings
    """
    
    # extract embeddings from manual annotation for the processed category_id
    df_xq, xq, ids_xq = initQuery(path_file_merged_xq,category_id) 
    
    # init index, database list, database ids and iterate over selected merged files
    index, xb, ids_xb = initIndex(gpu)   
    for path in path_list_merged_xb:
        # get df, remove categorized images
        df_xb = readDf(path)
        df_xb = cutTimestamp(df_xb,forbidden_timestamps)
        
        # apply different filters
        df_xb = filterAoI(df_xb,1) if AoI else df_xb
        df_xb = filterHead(df_xb,1) if head else df_xb
        df_xb = filterSolo(df_xb,1) if solo else df_xb
        
        # add all non-categorized embeddings to the index. 
        index,xb,ids_xb = augmentIndex(index,df_xb,xb,ids_xb)
    
    # stack (database, query) embeddings, and ids
    X = np.vstack(xb+[xq])
    ids_X = augmentIds(ids_xb,ids_xq)
    assert all([X.shape[0]==len(ids_X[k]) for k in range(ids_dim)])
    
    return index, X, ids_X, int(df_xq.shape[0])


def initCategoryIndex(path_dir_categorized, path_file_merged_xq, max_nb_instances=nb_augment_cat_index):
    """
    Init an index containing categorized embeddings. These embeddings come from the manual annotation tool 
    to which are added {max_nb_instances} categorized embeddings from each category, chosen at random.  
    Returns an index and the ids of the embeddings contained in the index.
    """
    # init index
    index_cat,xb_cat,ids_xb_cat = initIndex(gpu=False)   
    
    # extract embeddings from manual annotation and add manual annotations to index, since they have a category
    df_xq = readDf(path_file_merged_xq)
    index_cat,xb_cat,ids_xb_cat = augmentIndex(index_cat, df_xq, xb_cat, ids_xb_cat)
    
    # browse categorized embeddings. Each df corresponds to a different category
    for path in collectPaths(path_dir_categorized):
        if path and exists(path):
            df = readDf(path)
            df = df if df.shape[0]<= max_nb_instances else df.sample(max_nb_instances)
            index_cat, xb_cat, ids_xb_cat = augmentIndex(index_cat, df, xb_cat, ids_xb_cat)
    return index_cat, ids_xb_cat


def eggYolk(index_cat, ids_xb_cat:list, cat:int, X, new_nodes:list, cat_retrieved_nodes:set, kNN, k_out_of_N):
    """
    Perform the query {X[nodes]} on the {index_cat} to retrieve the {kNN} of each node.
    If {k_out_of_N} neighbors of a node are labeled as {cat}, the node is associated to the cat.
    """
    # query the kNN of each retrieved nodes, and look at the categories 
    _,I = index_cat.search(X[new_nodes,:],kNN)
    kNN_categories = filterIds(ids_xb_cat,I)[2]
    
    # categories of kNN should be identical to cat
    for node, category_votes in zip(new_nodes, kNN_categories):
        nb_match = (np.array(category_votes)==cat).sum()
        # collect node if k_out_of_kNN matches
        if nb_match >= k_out_of_N: cat_retrieved_nodes.add(node)
    return cat_retrieved_nodes


def updateCategorizedRows(points, channel, ids_xb, retrieved_nodes, category_id):
    """
    Fetch the rows from df_merged, adds a category_id and returns the 
    """
    
    categorized_rows = []
    
    # collect timestamps
    new_timestamps = set(filterIds(ids_xb, retrieved_nodes)[0])
    
    # load merged_dict and find merged files where new annotations have to be updated
    merged_dict = readJson(join(path_root_embeddings_from_masks,points,channel,'merged_dict.json'))
    merged_indices = sortedUnique([merged_dict[item] for item in new_timestamps])
    for merged_id in merged_indices:
        # load merged{merged_id}.csv
        path_merged = join(path_root_embeddings_from_masks,points,channel,'df_merged',f'merged{merged_id}.csv')
        df = readDf(path_merged)
        
        # collects timestamps belonging to the loaded df
        belongs_to = [item for item in new_timestamps if merged_dict[item]==merged_id]
        
        # map timestamps to nodes
        related_nodes = [item for item in retrieved_nodes if ids_xb[0][item] in belongs_to]
        
        # udpdate merged file
        for item in related_nodes:
            timestamp_id = ids_xb[0][item]
            SA_id = ids_xb[1][item]
            index_value = df[np.logical_and(df['timestamp_id']==timestamp_id, df['SA_id']==SA_id)].index
            df.loc[index_value,['category_id']]=category_id
            categorized_rows.append(df.loc[index_value])
            
    # all rows collected, merged the them together with the previous categorized rows
    path_file_categorized = join(path_root_embeddings_from_masks,points,channel,'df_categorized',f'{category_id}.csv')
    df_categorized = [readDf(path_file_categorized)] if exists(path_file_categorized) else []
    df_categorized += categorized_rows
    if df_categorized: writeDf(path_file_categorized, stackDfs(df_categorized) if len(df_categorized)>1 else df_categorized[0])


def filterIds(ids_xb,I,token=False):
    """
    Selects the ids corresponding to the nodes contained in I
    If I is an 2D array, the "ids per query" are returned
    """
    # container for the output
    ids = [[] for _ in range(ids_dim)]
    
    if type(I) is np.ndarray:
        nb_queries = I.shape[0]
        kNN = I.shape[1]
        I =  list(I.flatten())
        token = True
    elif type(I)==list or type(I)==set: I = [int(item) for item in I]
    
    elif type(I)==int: I = [I]
    
    # return [[timestamps],[SA_id],[cat]]
    for node in I:
        for idx in range(ids_dim):
            ids[idx] += [ids_xb[idx][node]]
    
    if token:
    # If I is a [2x3] 2D array, if corresponds to a request of finding the kNN = 3 nearest neighbors for nb_queries = 2 queries
    # [[a,b,c],[d,e,f]] ==> flattened ==> [a,b,c,d,e,f] ==> get ids ==> [[timestamps],[SA_ids],[category_ids]] ==> stack back ==> 
    # ==> [[[timestamps of q1], [timestamps of q2]],[[SA_ids of q1],[SA_ids of q2]],[[category_ids of q1],[category_ids of q2]]]
        for idx in range(ids_dim): 
            ids[idx] = [ids[idx][k*kNN:(k+1)*kNN] for k in range(nb_queries)]
    
    return ids


def augmentIds(ids_to_augment, ids_to_add):
    """
    Adds {ids_to_add} to {ids_to_augment}
    """
    for idx,items in enumerate(ids_to_add): 
        ids_to_augment[idx] += items
    return ids_to_augment


def getMaskQ(channel,ids_xq,idx):
    """
    The function fetches a mask from the manually annotated collection (Q for query)
    Returns the mask corresponding to the the {idx}-th (timestamp,SA_id) stored in {ids_xq} 
    """
    # ids 
    timestamp_id = ids_xq[0][idx]
    SA_id = ids_xq[1][idx]
    # path to masks file in "manual annotation directory"
    path = join(path_root_manual_annotation,channel,'ann_images',timestamp_id+'.json')
    
    return filterMasks(decodeMasks(readJson(path)),SA_id)


def getMaskB(points,channel,ids_xb,idx):
    """
    The function fetches a mask from the SA collection (B for dataBase)
    Returns the mask corresponding to the the {idx}-th (timestamp,SA_id) stored in {ids_xq} 
    """
    timestamp_id = ids_xb[0][idx]
    SA_id = ids_xb[1][idx]
    # path to masks file in "masks from images directory"
    path = join(path_root_masks_from_images,points,channel,timestamp_id+'.json')
    
    return filterMasks(decodeMasks(readJson(path)),SA_id)


