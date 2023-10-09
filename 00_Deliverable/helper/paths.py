from config_.paths import *

from helper.common_libraries import makedirs,listdir,exists,join,isfile

#**************************************************************************************************************
# retuns last element of a path
getName = lambda path: path.split('/')[-1]
# returns timestamp of a path
getId = lambda path: getName(path)[:23]
# returns channel of a path
getChannel = lambda path: getName(path)[:7]


def initPath(path):
    """
    Creates a directory located at {path}
    """
    if not exists(path): makedirs(path)
    return path


def searchRoot(root,excluded):
    """
    Explores a root and returns a list of new paths to files and a list of new discovered roots 
    """
    paths = []
    belongs_to_root = [root] if isfile(root) else [join(root,item) for item in listdir(root)]
    paths += [item for item in belongs_to_root if isfile(item) if not any([item.endswith(ext) for ext in excluded])]
    new_roots = [item for item in belongs_to_root if not isfile(item)]
    return paths,new_roots    


def collectPaths(roots, excluded = ['#','txt','db']):
    """
    Returns a list containing all paths to all files rooted at any root contained in the {roots} list input
    """
    if type(roots) == str: roots = [roots]
    paths=[]
    for root in roots:
        new_paths,new_roots = searchRoot(root,excluded=excluded)
        paths += new_paths
        roots += new_roots
    return sorted(paths)


def getImagePath(id,ext='.jpg'):
    """
    Returns the path to the image corresponding to the input {id}
    """
    splitted = id.split('_')
    channel = splitted[0]
    yyyy = splitted[1][:4]
    mm = splitted[1][4:6]
    dd = splitted[1][6:8]
    
    path_tail =  join(channel,yyyy,mm,dd,id+ext)
    
    return join(path_root_images_from_src, path_tail)


def alterPath(path_root_src, path_root_dest, path_file):
    '''
    path_file = path_root_src/.../any_file_name.any
    
    case imgs from video:
    path_root_src/channel/yyyy/mm/dd/image_id.jpg ==> 
    path_root_dest/channel/yyyy/mm/dd
    '''
    file_name = getName(path_file)
    path = path_file.replace(path_root_src, path_root_dest)
    path = path.replace(file_name,'')
    
    return initPath(path)



