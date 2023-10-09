from config_.paths import *
from config_.parameters import *

from helper.faiss import *
from helper.dijkstra import *
from helper.scriptIO import *
from helper.paths import getImagePath,initPath,getName
from helper.common_libraries import Image


def resize_and_copy_image(timestamp, cat, new_width=width_cutler):
    """
    Copies all images listed in the image_dict from the image database to a category folder.
    Resizes them to a width of {width_cutler} to be able to train CutLER using a batch of two images 
    without saturating the CUDA memory. 
    """
    path_dir_output = initPath(join(path_root_image_retrieval,timestamp[:7],str(cat)))
    image = Image.open(getImagePath(timestamp))
    width, height = image.size
    scaling_factor_x = width/new_width
    resized_image = image.resize((new_width,int(height/scaling_factor_x)),Image.LANCZOS)
    resized_image.save(join(path_dir_output,timestamp+'.jpg'))


if __name__ == "__main__":
    ############################################## CONFIG #### ###################################################################
    # masks&embeddings from channel _ , processed with _ points per side
    points,path_dir_src = requestPtsPerSide(path_root_embeddings_from_masks)
    channel,_ = requestChannel(path_dir_src)
    
    # paths to SAM database xb
    path_dir_emb_xb = join(path_root_embeddings_from_masks,points,channel,'df_merged')
    path_file_merged_dict = join(path_root_embeddings_from_masks,points,channel,'merged_dict.json')
    path_dir_categorized = initPath(join(path_root_embeddings_from_masks,points,channel,'df_categorized'))
    
    # paths to manual annotations xq
    path_dir_emb_xq = join(path_root_manual_annotation,channel,'df_merged')
    path_file_emb_xq = collectPaths(path_dir_emb_xq)[0]
    
    # path to the image dictionary, where are stored the timestamps of the found images
    path_file_image_dict = join(path_root_manual_annotation,channel,'image_dict.json')
    ############################################## RUN #######################################################################
    #**************************************** Global initialization **********************************************************
    # get available categories: the categories having manual annotations
    df_xq = readDf(path_file_emb_xq)
    available_categories = sortedUnique(extractValues(df_xq,'category_id'))
    
    # how many merged df are used to build the index. Each merged df has {nb_row_max} rows
    chunk = int(max_size_index/nb_rows_max)+1
    
    # container to store the retrieved images timestamps, for each available categories
    total_retrieved_images = [set() for _ in available_categories]
    
    for cat_idx,cat in enumerate(available_categories):
        print(f'\n########## Processing category {cat} ##########')
        path_file_categorized = join(path_dir_categorized,f'{cat}.csv')
        previously_found_timestamps = extractValues(readDf(path_file_categorized),'timestamp_id') if exists(path_file_categorized) else []
        round = 0
        while True:#***************************************** ROUND ***********************************************************
            print(f'\n***** Round {round} started *****')
            # container to store the images found during this round
            round_retrieved_images = set()
            
            # collect paths to merged file and shuffle them
            path_list_emb_xb = collectPaths(path_dir_emb_xb)
            random.shuffle(path_list_emb_xb)    
            
            # while some paths are available ==> run a sub-round using {chunk} merged_file only
            while path_list_emb_xb: 
                # init category index
                print('Building the category index...')
                index_cat, ids_xb_cat = initCategoryIndex(path_dir_categorized, path_file_emb_xq)
                
                # init main index 
                print('Building the index...')
                end_condition = len(path_list_emb_xb)<2*chunk
                path_list_selected = path_list_emb_xb if end_condition else path_list_emb_xb[:chunk] 
                path_list_emb_xb =  [] if end_condition else path_list_emb_xb[chunk:] 
                forbidden_timestamps = list(round_retrieved_images)+list(total_retrieved_images[cat_idx])+previously_found_timestamps+extractValues(filterCat(df_xq,cat),'timestamp_id')
                
                torch.cuda.empty_cache()
                index, X, ids_X, nb_annotations = initImageRetrieval(path_list_selected, path_file_emb_xq, cat, forbidden_timestamps)
                node_idx = list(np.arange(len(ids_X[0])))
                index_size = index.ntotal
                print(f'nb nodes in index: {index_size}')
                
                # init container for the retrieved nodes.
                retrieved_nodes = set()
                
                # build collection of graph and largest connected graph
                print('\nBuilding the main graph...')
                subgraphs = buildGraphCollection(index,X,kNN_graph)
                E0 = subgraphs[0][0]
                G0 = buildGraph(E0)
                nodes_of_main = [edge[0] for edge in E0]
                
                # run shortest-path
                print('Exploring the graph...')
                # the source of the path is always a manual annotations
                cat_manual_nodes = [item for item in node_idx[-nb_annotations:] if ids_X[2][item]==cat and str(item) in nodes_of_main]
                
                # for each combination (source, target), search for the shortest path [source ==> ... ==> target]
                for k,source in enumerate(cat_manual_nodes):
                    for target in cat_manual_nodes[k+1:]:
                        path,_ = shortestPath(G0,source,target)
                        # if there is at least a node on the way
                        if len(path)>2: 
                            # as the graph is undirected, it is possible to use a manually annotated node along the path. Cut them out
                            new_nodes = [int(item) for item in path[1:-1] if item not in node_idx[-nb_annotations:]]
                            
                            # run the eggYolk method on the new nodes to see if the reverse search confirms the similarity
                            retrieved_nodes = eggYolk(index_cat, ids_xb_cat, cat, X, new_nodes, retrieved_nodes, kNN_egg, k_out_of_kNN_egg)
                
                # save the timestamps for this category, for this sub-round
                new_timestamps = set(filterIds(ids_X, retrieved_nodes)[0])
                round_retrieved_images = round_retrieved_images.union(new_timestamps)
                
                # output sub-round information
                nb_new_images = len(new_timestamps)
                print(f'{cat} |nb nodes: {len(retrieved_nodes)} |nb images sub-round: {nb_new_images} |nb images round: {len(round_retrieved_images)} |nb images total: {len(total_retrieved_images[cat_idx])}')
                
                print('\nUpdating files...')
                updateCategorizedRows(points,channel,ids_X,retrieved_nodes,cat)               
            #********************************************* END OF SUB/ROUND ***********************************************************
            print(f'\n***** Round {round} ended *****')
            # gather retrieved timestamps and check how many images have been found during the round
            previous_nb_image = len(total_retrieved_images[cat_idx])
            total_retrieved_images[cat_idx] = total_retrieved_images[cat_idx].union(round_retrieved_images)
            nb_new_images = len(total_retrieved_images[cat_idx])-previous_nb_image
            print(f'{cat}: {nb_new_images} new images found over this round')
            
            # check whether too few/enough images were found
            if (round>=nb_min_round and nb_new_images<=nb_min_image): break
            if len(total_retrieved_images[cat_idx])>=nb_max_image: break
            
            round += 1
            if round == nb_max_round:break
    
    #**************************************************** ALL CATEGORIES EXPLORED ****************************************************
    # update the image_dict.
    # Using df_categorized instead of total_retrieved_images allows to add the images to the image_dict in the same order as the images were found.
    # The sooner it is found, the more likely it is to contain an instance of the category we are looking for. 
    print('Updating the image dict...')
    image_dict = readJson(path_file_image_dict)
    for path in collectPaths(path_dir_categorized):
        df = readDf(path)
        cat = getName(path).split('.')[0]
        list_timestamps = extractValues(df,'timestamp_id')
        image_dict[cat].update({'faiss':list_timestamps})
    writeJson(path_file_image_dict,image_dict)
    
    # copy images 
    print('\nCopying images...')
    for cat_idx, cat in enumerate(available_categories):
        if 'faiss' in image_dict[str(cat)].keys():
            timestamps = image_dict[str(cat)]['faiss']#+image_dict[str(cat)]['manual']
            [resize_and_copy_image(timestamp,cat) for timestamp in timestamps]
    
    # add a compact category dictionary to the image directory
    print('Adding a category dictionary...')
    channel_GT = readJson(join(path_root_GT,channel,f'{channel}_coco_annotations.json'))
    category_dict = {item['id']:item['name'] for item in channel_GT['categories'] if item['id'] in available_categories}
    writeJson(join(path_root_image_retrieval,channel,'category_dict.json'),category_dict)
