from helper.tuning import imreadRGB
from helper.lib_detectron2 import getCatalog, Visualizer
from helper.common_libraries import join,plt

from maskcut.colormap import random_color, random_colors


def visualizePseudoGT(sample, catalog_name_previous, catalog_name_new, p=15, path_save=False):
    '''
    Creates a 1x3 plot diplaying:
    i. the image without annotations
    ii. the image with annotation of catalog named {catalog_name_previous}
    iii. the image with annotation of catalog named {catalog_name_new}
    if "path_save" parameter is an existing path, the plot is saved at the given location
    '''
    
    # get catalogs
    catalog_names = [catalog_name_previous, catalog_name_new]
    catalogs = []
    metas = []
    for catalog_name in catalog_names:
        catalog,meta = getCatalog(catalog_name)
        catalogs.append(catalog)
        metas.append(meta)
    
    # get sample name            
    file_name = catalogs[0][sample]['file_name'].split('/')[-1]
    
    # read image
    image = imreadRGB(catalogs[0][sample]['file_name'])
    visualizer = []
    out = []
    for c,m in zip(catalogs,metas):
        visualizer.append(Visualizer(image, metadata=m, scale=0.5))
        out.append(visualizer[-1].draw_dataset_dict(c[sample]).get_image()) 
    
    # Define the size of the figure
    fig, axs = plt.subplots(1, 3, figsize=(p, 3*p))
    axs = axs.flatten()
    for i, ax in enumerate(axs):
        if i==0:
            ax.imshow(image)
            ax.set_title(f'input: {file_name}')
        elif i <= len(out):
            ax.imshow(out[i-1])
            ax.set_title(catalog_names[i-1])
        # If there are no more images, turn off the axis to leave it blank
        else:
            ax.axis("off")
            
    if path_save!=False: plt.savefig(join(path_save,f'{file_name}_visu.jpg'), bbox_inches='tight')
    
    plt.tight_layout()
    plt.show()