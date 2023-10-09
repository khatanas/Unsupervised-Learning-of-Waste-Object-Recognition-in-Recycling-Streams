from helper.paths import collectPaths,initPath,getName,getId
from helper.common_libraries import shutil,subprocess,sys,torch,os,join

def copyToTmp(path_list_file,path_dir_tmp):
    """
    Creates a tmp directory
    Copies files in path_list_file to tmp directory
    """
    path_dir_tmp = initPath(path_dir_tmp)
    for path in path_list_file:
        shutil.copy(path,join(path_dir_tmp,getName(path)))
        
def moveToMain(path_dir_tmp, path_dir_main):
    """
    Deletes all files from main directory sharing their name with any file in tmp directory
    Moves the files from tmp directory to main directory (update)
    Deletes tmp directory
    """
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
        
    # init threads: compute batch_size, and create a tmp directory name for each threads
    batch_size = int(len(path_list_masks_filtered)/nb_threads)
    path_list_dir_batch = [join(path_dir_src_masks,str(i).zfill(2)) for i in range(nb_threads)]
    
    processes = []
    for thread_id in range(nb_threads):
        # pick tmp directory, split masks list
        path_dir_batch = path_list_dir_batch[thread_id]
        path_list_batch = path_list_masks_filtered[thread_id*batch_size:(thread_id+1)*batch_size] if thread_id<nb_threads-1 else path_list_masks_filtered[thread_id*batch_size:]
        copyToTmp(path_list_batch, path_dir_batch)
        
        if path_list_batch:
            embeddings_process = subprocess.Popen(['python', 'sub_embeddings.py', path_dir_batch, path_dir_output_emb,str(-1)])
            processes.append(embeddings_process)
        
    for k,process in enumerate(processes):
        if process.poll() is None: process.wait() 
        
    # delete original files and move cleaned masks. Delete tmp directories
    for path_dir_batch in path_list_dir_batch: 
        moveToMain(path_dir_batch, path_dir_src_masks)
