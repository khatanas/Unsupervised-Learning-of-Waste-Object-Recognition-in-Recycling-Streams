import os
from os.path import join, exists
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shutil

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
    
    
def plotDPIthInfo(p2dpith,rnd,DPIs,ths,dataset_test,json_path_GT_te):
    nb_max = 0
    
    # GET GT INFO
    with open(json_path_GT_te, 'r') as f:
        tmp_GT = json.load(f)
    # list of areas
    vals_GT = [d['bbox'][2]*d['bbox'][3] for d in tmp_GT['annotations']]
    # avg nb_annotation
    nb_GT =len(vals_GT)/len(tmp_GT['images'])
    # median of area
    GT_median = np.median(vals_GT)
            
    # GET param combination INFO
    tmp_areas = []
    for dpi in DPIs:
        for tmp_th in ths:
            # compose name and open file
            tmp_rnd_name = f'te_{dataset_test}_rd{rnd}_DPI{dpi}_th{tmp_th}.json'
            tmp_new = join(p2dpith,tmp_rnd_name)
            with open(tmp_new, 'r') as f:
                lala = json.load(f)
                
            tmp_ann = [d['bbox'][2]*d['bbox'][3] for d in lala['annotations']]
            tmp_nb = len(tmp_ann)/len(lala['images'])
            
            # add header containing serie data parameters
            tmp_data = [dpi,tmp_th,tmp_nb]+tmp_ann
            if len(tmp_data)>nb_max: nb_max=len(tmp_data)
            tmp_areas.append(tmp_data)
        
        # add header containing GT parameters
        tmp_data = [dpi,'GT',nb_GT]+vals_GT
        if len(tmp_data)>nb_max: nb_max=len(tmp_data)
        tmp_areas.append(tmp_data)

    print(nb_max)
    padded = [
        tmp_areas[idx]+f*[None]
        for idx,f in enumerate(
            [nb_max-len(l) for l in tmp_areas] 
        )
    ]

    df = pd.DataFrame(data=padded, columns=['DPI','th','nb']+[str(i) for i in range(nb_max-3)])
    df.to_csv(join(p2dpith,'df_dpith.csv'), index=False)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.barplot(x='DPI', y='value', hue='th', data=pd.melt(df, id_vars=['DPI', 'th'], value_vars=df.columns[2]))
    plt.ylabel('nb_mask/image (barplot)')
    plt.axhline(y=nb_GT, color='pink', linestyle='--')

    plt.title(f'Prediction info (rd{rnd}): {dataset_test}')
    plt.legend(loc='center right', bbox_to_anchor=(1.25, 0.5))

    scatter_ax = ax.twinx()
    sns.boxplot(x='DPI', y='value', hue='th', data=pd.melt(df, id_vars=['DPI', 'th'], value_vars=df.columns[3:]))
    plt.yscale('log')
    plt.ylabel('bbox_area [pixels] (boxplot)')
    plt.axhline(y=GT_median, color='orange', linestyle='--')

    scatter_ax.get_legend().remove()
    plt.savefig(join(p2dpith,f'dpith.png'),bbox_inches='tight')


def gatherCSVs(root_train,rnd,offset=0):
    tmp_dest_loc = join(root_train,'gathered')
    os.makedirs(tmp_dest_loc, exist_ok=True)
    start_=1 if offset==0 else offset
    for i in range(start_,rnd+1):
        tmp_src_loc = join(root_train,f'rd{i-1}-{i}',f'eval_{i-1}to{i}')
        for j in ['bbox', 'segm']:
            tmp_src = join(tmp_src_loc,f'{j}.csv')
            tmp_dest = join(tmp_dest_loc,f'{j}{i-1}-{i}.csv')
            shutil.copyfile(tmp_src, tmp_dest)
            
        tmp_src_loc = join(root_train,f'rd{i-1}-{i}')
        for j in ['tr', 'tr']:
            tmp_src = join(tmp_src_loc,f'{j}_jerry_real_rd{i}_DPI50_th0.95.json')
            tmp_dest = join(tmp_dest_loc,f'{j}_jerry_real_rd{i}_DPI50_th0.95.json')
            if exists(tmp_src): shutil.copyfile(tmp_src, tmp_dest)
            else:continue
            
def mergeCSVs(root_train,rnd,iter_inter,offset=0):
    tmp_loc = join(root_train,'gathered')
    for j in ['bbox','segm']:
        start_=1 if offset==0 else offset
        for i in range(start_,rnd+1):
            if i==start_:
                pre = pd.read_csv(join(tmp_loc,f'{j}{i-1}-{i}.csv'))
            else:
                curr = pd.read_csv(join(tmp_loc,f'{j}{i-1}-{i}.csv'))
                pre = pd.concat([pre[:-1],curr[1:]],axis=0)
                
        pre.reset_index(drop=True,inplace=True)
        df_merge = pre[:-1]
        df_merge['CP_nb'] = [0]+[i*iter_inter-1 for i in range(1,df_merge.shape[0])]
        df_merge.to_csv(join(tmp_loc,f'{j}_merged.csv'), index=False)
        
def plotCSVs(root_train,rnd,offset=0,iter_event=False):
    tmp_loc = join(root_train,'gathered')
    tmp_bbox = pd.read_csv(join(tmp_loc,'bbox_merged.csv'))
    tmp_segm = pd.read_csv(join(tmp_loc,'segm_merged.csv'))
    name = root_train.split('/')[-1]
    subnames = ['bbox','segm']

    fig, axes = plt.subplots(2, 1, figsize=(12, 10))  # Create a figure with two subplots
    for idx, df in enumerate([tmp_bbox,tmp_segm]):
        ax = axes[idx]  # Select the appropriate subplot

        [ax.plot(df['CP_nb'], df[col], label=col) for col in df.columns[:-1]]
        ax.set_ylabel('%')
        ax.set_title(f'Test info ({offset} to rd{rnd}): {name} - {subnames[idx]} ')
        ax.legend(loc='upper left')
        
        if iter_event != False:
            for i in iter_event:
                ax.axvline(x=i, color='k', linestyle='-.')

    ax.set_xlabel('iteration')
    plt.savefig(join(tmp_loc, 'combined_plots.png'),bbox_inches='tight')
    plt.show()
    
