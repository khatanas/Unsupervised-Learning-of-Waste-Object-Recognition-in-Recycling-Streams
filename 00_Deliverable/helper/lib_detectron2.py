from helper.tuning import imreadRGB
from helper.common_libraries import plt, join

from detectron2.utils.visualizer import Visualizer
from detectron2.data.catalog import MetadataCatalog, DatasetCatalog
from detectron2.data.datasets import register_coco_instances



def registerCatalog(catalog_name, path_file_json, path_dir_images):
    '''
    Creates a detectron2 catalog called {catalog_name} gathering info
    from json located at path_file_json and images located at  path_dir_images
    '''
    if catalog_name in DatasetCatalog.list():
        DatasetCatalog.remove(catalog_name)
        MetadataCatalog.remove(catalog_name)
    register_coco_instances(catalog_name, {}, path_file_json,  path_dir_images)
    print(f'registered: {catalog_name}\njson is located at: {path_file_json}\nimages are located at: { path_dir_images}\n')


def getCatalog(catalog_name):
    catalog = DatasetCatalog.get(catalog_name)
    meta = MetadataCatalog.get(catalog_name)
    return catalog,meta


def visualizeD2(sample_idx, catalog_name, title='',axis='off', path_save=False):
    '''
    Outputs the {sample_idx}-th image of the catalog {catalog_name} together with its annotations 
    '''
    catalog,meta = getCatalog(catalog_name)
    sample = catalog[sample_idx]
    file_name = sample['file_name'].split('/')[-1]
    count = len(sample["annotations"])
    
    # use detectron2 Visualizer
    image = imreadRGB(sample["file_name"])
    visualizer = Visualizer(image, metadata=meta, scale=0.5)
    out = visualizer.draw_dataset_dict(sample).get_image()
    
    plt.figure()
    if title != '': 
        plt.title(f'{title}\n {count} annotations')
    plt.imshow(out)
    plt.axis(axis)
    if path_save!=False: plt.savefig(join(path_save,f'{file_name}\n{count} annotations.jpg'),bbox_inches='tight')
    plt.show()