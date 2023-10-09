from helper.tuning import imreadRGB
from helper.lib_detectron2 import getCatalog, Visualizer
from helper.common_libraries import join,plt,pd,json

from maskcut.colormap import random_color, random_colors

#**************************************************************************************************************
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


def init_df(path2logfile):
    '''
    CutLER helper: 
    Initialize dataframe to collect AP output
    '''
    
    # load log.txt
    with open(join(path2logfile,'log.txt')) as f:
        text = f.read()
    
    keep = [row for row in text.split('\n') if 'd2.evaluation.testing' in row][-6:]
    names = keep[1].split('copypaste: ')[1].split(',')
    names.append('CP_nb')
    df = pd.DataFrame(columns=names)
    
    return df

def get_results(path2logfile, df_bbox, df_segm, nb):
    '''
    CutLER helper: 
    Add AP output to initialized dataframe
    ''' 
    # load log.txt
    with open(join(path2logfile,'log.txt')) as f:
        text = f.read()
        
    # split into rows and keep lines containing evaluation results 
    keep = [row for row in text.split('\n') if 'd2.evaluation.testing' in row][-6:]
    vals = [keep[idx].split('copypaste: ')[1].split(',') for idx in [2,5]]
    
    vals_bbox = [float(d) if d != 'nan' else 0 for d in vals[0]]
    vals_bbox.append(nb)
    df_bbox.loc[len(df_bbox)] = vals_bbox
    
    vals_segm = [float(d) if d != 'nan' else 0 for d in vals[1]]
    vals_segm.append(nb)
    df_segm.loc[len(df_bbox)] = vals_segm


def plotTrainingInfo(p2o,p2t,rnd, dataset_train):
    count = 0
    with open(join(p2o,'metrics.json'), 'r') as f:
        # Read the contents of the file
        for line in f:
            count +=1
            
    with open(join(p2o,'metrics.json'), 'r') as f:
        lala = [json.loads(next(f)) for _ in range(count)]
        
    df_train = {
        'total_loss':[f['total_loss'] for f in lala if 'lr' in f.keys()],
        'lr':[f['lr'] for f in lala if 'lr' in f.keys()],
        'iteration':[f['iteration'] for f in lala if all(key in f.keys() for key in ['lr','iteration'])]
        }
    
    df_train = pd.DataFrame(data=df_train)
    df_train.to_csv(join(p2t,'train.csv'), index=False)
    
    # Create a figure and axis objects
    fig, ax1 = plt.subplots()
    
    # Create scatter plots for y1 and y2 on ax1
    ax1.plot(df_train['iteration'], df_train['total_loss'], label='loss')
    ax1.set_xlabel('iteration')
    ax1.set_ylabel('total loss')
    ax1.set_title(f'Training info (rd{rnd-1} to {rnd}): {dataset_train} ')
    
    # Create a twin axis for y3
    ax2 = ax1.twinx()
    ax2_color = 'r'
    ax2.plot(df_train['iteration'], df_train['lr'], label='lr', color=ax2_color)
    
    # Set label for ax2
    ax2.set_ylabel('lr', color=ax2_color)
    
    # Display the plot
    plt.savefig(join(p2t,f'train.png'),bbox_inches='tight')
    plt.show()
    
def plotTestInfo(p2t,rnd,dataset_test):
    df_bbox = pd.read_csv(join(p2t,'bbox.csv'))
    df_segm = pd.read_csv(join(p2t,'segm.csv'))

    dfs = [df_bbox, df_segm]
    names = ['bbox','segm']
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))  # Create a figure with two subplots

    for idx, df in enumerate(dfs):
        ax = axes[idx]  # Select the appropriate subplot

        [ax.plot(df['CP_nb'], df[col], label=col) for col in df.columns[:-1]]
        ax.set_xlabel('iteration')
        ax.set_ylabel('%')
        ax.set_title(f'Test info (rd{rnd-1} to {rnd}): {dataset_test} - {names[idx]} ')
        ax.legend()

    plt.savefig(join(p2t, 'combined_plots.png'),bbox_inches='tight')
    plt.show()