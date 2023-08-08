from config.paths import *

from helper.common_libraries import makedirs,listdir,exists,join,isfile,getsize


getName = lambda path: path.split('/')[-1]    
getId = lambda path: getName(path)[:23]
getChannel = lambda path: getName(path)[:7]


def describePath(path, description=''):
    files = collectPaths(path) 
    print(f'{description}: {len(files)} files, {int(sum([getsize(file)/(1024**3) for file in files]))} GB')


def initPath(path):
    if not exists(path): makedirs(path)
    return path


def searchRoot(root,excluded):
    paths = []
    belongs_to_root = [root] if isfile(root) else [join(root,item) for item in listdir(root)]
    paths += [item for item in belongs_to_root if isfile(item) if not any([item.endswith(ext) for ext in excluded])]
    new_roots = [item for item in belongs_to_root if not isfile(item)]
    return paths,new_roots    


def collectPaths(roots, excluded = ['#','txt','db']):
    if type(roots) == str: roots = [roots]
    paths=[]
    for root in roots:
        new_paths,new_roots = searchRoot(root,excluded=excluded)
        paths += new_paths
        roots += new_roots
    return sorted(paths)


def getTailPath(id,ext='.jpg'):
    """
    id.ext ==> channel/yyyy/mm/dd/id.ext
    """
    splitted = id.split('_')
    channel = splitted[0]
    yyyy = splitted[1][:4]
    mm = splitted[1][4:6]
    dd = splitted[1][6:8]
    
    return join(channel,yyyy,mm,dd,id+ext)


def getImagePath(id,ext='.jpg'):
    """
    Return path_file_image
    """
    return join(path_root_images_from_src, getTailPath(id,ext=ext))


def alterPath(path_root_src, path_root_dest, path_file):
    '''
    path_file_any = path_root_src/.../any_file_name.any
    
    case imgs from video:
    path_root_src/channel/yyyy/mm/dd/image_id.jpg ==> 
    path_root_dest/channel/yyyy/mm/dd
    '''
    file_name = getName(path_file)
    path = path_file.replace(path_root_src, path_root_dest)
    path = path.replace(file_name,'')
    
    return initPath(path)
