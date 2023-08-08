from config.paths import *
from config.parameters import emb_dim,ids_dim

from helper.dataframes import *
from helper.paths import collectPaths
from helper.annotations import decodeMasks
from helper.tuning import readJson,sortedUnique, flattenList
from helper.common_libraries import join,np,torch

import faiss
import networkx as nx
from heapq import heappop, heappush


newTimestamps = lambda ids_xb,I: sortedUnique([ids_xb[0][k] for k in list(I.flatten())])


def initFaissIndex(gpu):
    index = faiss.IndexFlatL2(emb_dim)
    return faiss.index_cpu_to_gpu(faiss.StandardGpuResources(),0,index) if (torch.cuda.is_available() and gpu) else index


def addFaissIndex(index,df,ids):
    # collect info for mapping
    ids[0]+=extractValues(df,'timestamp_id')
    ids[1]+=extractValues(df,'SA_id')
    ids[2]+=extractValues(df,'category_id')
    # add embeddings to index
    index.add(extractEmbeddings(df))
    return index,ids


def initQuery(path_dir_merged,category_id=False):
    # get df
    df_xq = readDf(collectPaths(path_dir_merged)[0])
    # filter category id if requested
    if category_id: df_xq = filterCat(df_xq,category_id)
    # extract embeddings, timestamp_id, SA_id from df
    xq = extractEmbeddings(df_xq)
    ids_xq = extractIds(df_xq)
    return df_xq, xq, ids_xq


def initQuickInstanceRetrieval(path_dir_merged_xb, path_dir_merged_xq, category_id, gpu=False):
    # init index
    index = initFaissIndex(gpu)
    
    # extract embeddings from manual annotation
    _,xq, ids_xq = initQuery(path_dir_merged_xq,category_id)
    
    # init embeddings mapping
    ids_xb = [[] for _ in range(ids_dim)]
    
    # iterate over collection of merged files
    for path in collectPaths(path_dir_merged_xb):
        # get df, keep masks with AoI==1, keep masks being "head"
        df_xb = readDf(path)
        df_xb = filterAoI(df_xb,1)
        df_xb = filterHead(df_xb,1)
        
        # remove all masks coming from seen image (the goal is to find new images...)
        df_xb = filterTimestamp(df_xb,ids_xq[0])
        
        # augment index
        index,ids_xb = addFaissIndex(index,df_xb,ids_xb)
        
    # ==> returns the index, the queries embeddings, the index mapping
    return index, xq, ids_xb


def initImageRetrieval(path_dir_merged_xb, path_dir_merged_xq, gpu=True):
    # init index
    index = initFaissIndex(gpu)
    
    # extract embeddings from manual annotation
    df_xq, xq, ids_xq = initQuery(path_dir_merged_xq)
    
    # add manual annotations to index, requested for building a graph with ease
    index,ids_xb = addFaissIndex(index, df_xq, [[] for _ in range(ids_dim)])
    
    # collect embeddings
    X = [xq]
    
    # iterate over collection of merged files
    for path in collectPaths(path_dir_merged_xb):
        # get df, keep masks with AoI==1, keep masks being "head"
        df_xb = readDf(path)
        df_xb = filterAoI(df_xb,1)
        df_xb = filterHead(df_xb,1)
        
        # add categorized embeddings to index, requested for building a graph with ease
        df_categorized = df_xb[df_xb['category_id']!=-1]      
        timestamps_to_filter_out = ids_xq[0] + extractValues(df_categorized,'timestamp_id')
        X.append(extractEmbeddings(df_categorized))
        index,ids_xb = addFaissIndex(index,df_categorized,ids_xb)
        
        # add all non-categorized embeddings, except those from see images (the goal is still to find new images...)
        df_xb = filterTimestamp(df_xb,timestamps_to_filter_out)
        X.append(extractEmbeddings(df_xb))
        index,ids_xb = addFaissIndex(index,df_xb,ids_xb)
        
    X = np.vstack(X)
    return index, X, ids_xb, df_xq.shape[0]


def initCategoryIndex(path_dir_merged_xb,path_dir_merged_xq, gpu=False):
    # init index
    index = initFaissIndex(gpu)
    
    # extract embeddings from manual annotation
    df_xq,_,_ = initQuery(path_dir_merged_xq)
    
    # add manual annotations to index, since they have a category
    index,ids_xb = addFaissIndex(index, df_xq, [[] for _ in range(ids_dim)])
    
    # iterate over collection of merged files
    for path in collectPaths(path_dir_merged_xb):
        # get df, keep masks with AoI==1, keep masks being "head"
        df_xb = readDf(path)
        df_xb = filterAoI(df_xb,1)
        df_xb = filterHead(df_xb,1)
        
        # keep these with a category
        df_xb = df_xb[df_xb['category_id']!=-1]        
        
        # augment index
        index,ids_xb = addFaissIndex(index,df_xb,ids_xb)
    return index, ids_xb


def collectEdges(index,X,k):
        # find the (k+1)-th nearest neighbor of each mask (1st is self)
    E = []
    for idx in range(index.ntotal):
        if idx%1000==0:print(f'{idx}/{index.ntotal}')
        D,I = index.search(X[idx:idx+1,:], k+1)
        D = list(D.squeeze())
        I = list(I.squeeze())
        # get rid of first retrieval and store remaining as edge: [u,v,weight]
        for k_th in range(1,k+1): E.append([str(idx),str(I[k_th]),D[k_th]])
    return E


def buildGraph(E):
    # build undirected weighted graph
    G = nx.Graph()
    G.add_edges_from([(edge[0],edge[1],{'weight':edge[2]}) for edge in E])
    return G


def buildGraphCollection(index,X,k):
    # collect edges
    E = collectEdges(index,X,k)
    
    nb_edge = len(E)
    nb_node = index.ntotal
    
    # find disconnected graphs
    groups_of_nodes = [list(vertex) for vertex in nx.connected_components(buildGraph(E))]
    nb_graph = len(groups_of_nodes)
    
    print(f'\nThe main graph is connected') if nb_graph==1 else print(f'\nThe main graph is made of {nb_graph} disconnected graphs:')
    
    # build a dictionary to assign a subgraph to each node (speed up edge filtering)
    belongs_to = {node:-1 for node in range(nb_node)}
    for graph_id in range(nb_graph):
        for node_id in groups_of_nodes[graph_id]:
            belongs_to[node_id] = graph_id
            
    # init list to store filtered vertices, subgraphs, and SCCs 
    subgraphs = [[] for _ in range(3)]
    for graph_id in range(nb_graph):
        # filter edges
        subgraphs[0].append([E[edge_id] for edge_id in range(nb_edge) if belongs_to[E[edge_id][0]]==graph_id])
        # build unweighted directed subgraph
        subG = nx.DiGraph()
        subG.add_edges_from([(edge[0],edge[1]) for edge in subgraphs[0][graph_id]])
        subgraphs[1].append(subG)
        # find SCC of subgraph with cardinality>1
        subgraphs[2].append([nodes for nodes in list(nx.strongly_connected_components(subgraphs[1][graph_id])) if len(nodes)>1])
        
        flattened = [node for group in subgraphs[2][graph_id] for node in group]
        print(f'Subgraph {graph_id}: {len(groups_of_nodes[graph_id])} vertices, with {len(subgraphs[2][graph_id])} SCC, totalizing {len(flattened)} vertices')
    
    return subgraphs


def shortestPath(G,src,target,forbidden_nodes=[]):
    # graph nodes are of type str
    src = str(src)
    target = str(target)
    forbidden_nodes = [str(item) for item in forbidden_nodes]
    
    # init dict for distances and path retrieval
    parents = {node: -1 for node in G.nodes()}
    distances = {node: float('inf') for node in G.nodes()}
    distances[src] = 0
    
    # assess existence of target in graph
    if target not in distances.keys(): 
        print(f'{target} not in G')
        return [[] for _ in range(2)]
    
    # init searching 
    visited = set()
    heap = [(0, src)]
    while heap:
        _,current_node = heappop(heap)
        if current_node == target: break
        if current_node not in visited:
            visited.add(current_node)
            # iterate over current node's neighbors
            for neighbor, attributes in G[current_node].items():
                # check if neighbor is not forbidden
                distance_step = float('inf') if neighbor in forbidden_nodes else attributes['weight']
                # update distance and parent if better path is found
                if distances[current_node] + distance_step < distances[neighbor]:
                    distances[neighbor] = distances[current_node] + distance_step
                    parents[neighbor] = current_node
                    heappush(heap,(distances[neighbor],neighbor))
    
    if distances[target]<float('inf'):
        # retrieve path src ==> dest
        path = [target]
        while path[0]!=src: path = [parents[path[0]]] + path  
        # collect distances 
        dists = [distances[node] for node in path]
        
        return path,dists
    
    else:
        print(f'{src} and {target} disconnected')
        return [[] for _ in range(2)]


def getMaskQ(channel,ids_xq,idx):
    # ids 
    timestamp_id = ids_xq[0][idx]
    SA_id = ids_xq[1][idx]
    # path to masks file in "manual annotation directory"
    path = join(path_root_manual_annotation,channel,'ann_images',timestamp_id+'.json')
    # masks
    masks = decodeMasks(readJson(path))
    # specific mask
    mask = [mask for mask in masks if mask['id']==SA_id]
    return mask[0] if len(mask)==1 else []


def getMaskB(points,channel,ids_xb,idx):
    timestamp_id = ids_xb[0][idx]
    SA_id = ids_xb[1][idx]
    # path to masks file in "masks from images directory"
    path = join(path_root_masks_from_images,points,channel,timestamp_id+'.json')
    # masks
    masks = decodeMasks(readJson(path))
    # specific masks
    mask = [mask for mask in masks if mask['id']==SA_id]
    return mask[0] if len(mask)==1 else []


def updateCategories(points, channel, ids_xb, retrieved_nodes, available_categories):
    # flatten the nested lists
    flatten_nodes = flattenList(retrieved_nodes)
    
    # build a category dictionary
    category_dict = {node:-1 for node in flatten_nodes}
    for cat_idx, cat in enumerate(available_categories):
        for node in retrieved_nodes[cat_idx]:
            category_dict[node]=cat
    
    # collect timestamps and unique timestamps
    all_timestamps = [ids_xb[0][node] for node in flatten_nodes]
    unique_timestamps = sortedUnique(all_timestamps)
    
    # load merged_dict and iterate over merged_ids
    merged_dict = readJson(join(path_root_embeddings_from_masks,points,channel,'merged_dict.json'))
    for merged_id in sortedUnique(merged_dict.values()):
        # load merged{merged_id}.csv
        path = join(path_root_embeddings_from_masks,points,channel,'df_merged',f'merged{merged_id}.csv')
        df = readDf(path)
        
        # collects timestamps belonging to the loaded df
        belongs_to = [timestamp for timestamp in merged_dict.keys() if merged_dict[timestamp]==merged_id and timestamp in unique_timestamps]
        
        # map timestamps to nodes
        related_nodes = [node for node in flatten_nodes if ids_xb[0][node] in belongs_to]
        
        # udpdate merged file
        for node in related_nodes:
            timestamp_id = ids_xb[0][node]
            SA_id = ids_xb[1][node]
            index_value = df[np.logical_and(df['timestamp_id']==timestamp_id, df['SA_id']==SA_id)].index
            df.loc[index_value,['category_id']]=category_dict[node]
        # save updates
        writeDf(path,df)


def filterIds(ids_xb,I,token=False):
    ids = [[] for _ in range(ids_dim)]
    if type(I) is np.ndarray:
        nb_query = I.shape[0]
        kNN = I.shape[1]
        I =  list(I.flatten())
        token = True
    elif type(I) == list: I = [int(item) for item in I]
    elif type(I) == int or type(I) == str: I = [int(I)]
        
    for k in I:
        for idx in range(ids_dim):
            ids[idx] += [ids_xb[idx][k]]
    
    if token:
        for idx in range(ids_dim): 
            ids[idx] = [ids[idx][k*kNN:(k+1)*kNN] for k in range(nb_query)]
    return ids