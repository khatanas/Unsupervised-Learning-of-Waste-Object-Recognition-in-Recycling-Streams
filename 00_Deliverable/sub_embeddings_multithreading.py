from helper.paths import collectPaths,initPath,getName,getId
from helper.common_libraries import shutil,subprocess,sys,torch,os,join

def copyToTmp(path_list_file,path_dir_tmp):
    path_dir_tmp = initPath(path_dir_tmp)
    for path in path_list_file:
        shutil.copy(path,join(path_dir_tmp,getName(path)))

def moveToMain(path_dir_tmp, path_dir_main):
    for path in collectPaths(path_dir_tmp):
        path_dest = join(path_dir_main,getName(path))
        os.remove(path_dest)
        shutil.move(path,path_dest)
    os.rmdir(path_dir_tmp)

if __name__ == "__main__":
    torch.cuda.empty_cache()
    os.environ["MKL_THREADING_LAYER"] = "GNU"
    ############################################## CONFIG #######################################################################
    path_dir_src_masks = sys.argv[1]
    path_dir_output_emb = sys.argv[2]
    nb_threads = int(sys.argv[3])
    
    # remaining masks to process
    path_list_masks = collectPaths(path_dir_src_masks)
    existing_embeddings_timestamp = [getId(path) for path in collectPaths(path_dir_output_emb)]
    path_list_masks_filtered = [path for path in path_list_masks if getId(path) not in existing_embeddings_timestamp]
    
    # init threads
    batch_size = int(len(path_list_masks_filtered)/nb_threads)
    path_list_dir_batch = [join(path_dir_src_masks,str(i).zfill(2)) for i in range(nb_threads)]
    
    for thread_id in range(nb_threads):
        # pick tmp directory, split masks list
        path_dir_batch = path_list_dir_batch[thread_id]
        path_list_batch = path_list_masks_filtered[thread_id*batch_size:(thread_id+1)*batch_size] if thread_id<nb_threads-1 else path_list_masks_filtered[thread_id*batch_size:]
        copyToTmp(path_list_batch, path_dir_batch)
        
        # launch thread
        if len(path_list_batch)==batch_size and thread_id<nb_threads-1:
            subprocess.Popen(['python', 'sub_embeddings.py', path_dir_batch, path_dir_output_emb,str(-1)])
        
        # launch final thread
        else: subprocess.run(['python', 'sub_embeddings.py', path_dir_batch, path_dir_output_emb,str(-1)])   
        
    # delete original files and move cleaned masks. Delete tmp directories
    for path_dir_batch in path_list_dir_batch: 
        moveToMain(path_dir_batch, path_dir_src_masks)
        
        
