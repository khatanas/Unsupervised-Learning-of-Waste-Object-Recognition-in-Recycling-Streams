import sys
sys.path.append('../')

from config.paths import *
from config.parameters import crop_offset

from helper.annotations import decodeMasks,cropMask2
from helper.paths import getImagePath
from helper.dataframes import extractIds
from helper.tuning import readJson,imreadRGB
from helper.faiss import getMaskB, getMaskQ,shortestPath
from helper.common_libraries import cv2,plt,np,sns

#*************************** 00 *************************************
def get_video_frame_format(video_file_path):
    cap = cv2.VideoCapture(video_file_path)
    if not cap.isOpened():
        print("Error: Unable to open the video file.")
        return
    ret, frame = cap.read()
    if not ret:
        print("Error: Unable to read the first frame.")
        cap.release()
        return
    frame_format = frame.shape
    print(f"Frame format: {frame_format}")
    cap.release()
    
    
#************************** 03 **************************************
def visualizeQuery(channel,points,
                    df_xq,
                    ids_xb,D,I,
                    n_row=3, p=2):
    
    n_row = min(n_row,df_xq.shape[0])
    n_col = I.shape[1]+1
    ids_xq = extractIds(df_xq)
    
    fig, axes = plt.subplots(n_row, n_col, figsize=((n_col)*p,n_row*p))
    for row in range(n_row):
        for col in range(n_col):
            mask = getMaskQ(channel,ids_xq,row) if col == 0 else getMaskB(points,channel,ids_xb,I[row,col-1])
            axes[row, col].imshow(cropMask2(mask,crop_offset))
            axes[row, col].axis('off')
    
    [axes[0, col].set_title(title) for col,title in enumerate(['Query']+[f'Result {k+1}' for k in range(n_col-1)])]
    [axes[row, 0].text(-0.15, 0.5, f'{row}', transform=axes[row, 0].transAxes, fontsize=12, rotation=90, va='top') for row in range(n_row)]
    
    plt.tight_layout()
    plt.show()
    
    # Create the heatmap using seaborn
    plt.figure(figsize=((n_col-1)*p, n_row*p/2))
    sns.heatmap(D[:n_row,:], annot=True, cmap='Blues', xticklabels=np.arange(1,n_col), yticklabels=np.arange(n_row))
    
    # Add labels and title
    plt.xlabel('k-th neighbor')
    plt.ylabel('Query id')
    plt.title(f'L2 distances')
    
    # Display the plot
    plt.show()


def visualizeBranches(points,channel,I,ids_xb,from_query=True, n_row=3, p=2):
    n_row = min(n_row,I.shape[0])
    n_col = I.shape[1]
    
    fig, axes = plt.subplots(n_row, n_col, figsize=((n_col)*p,n_row*p))
    
    if n_row >1:
        for row in range(n_row):
            for col in range(n_col):
                if I[row,col] >=0:
                    mask = getMaskQ(channel,ids_xb,I[row,col]) if (col==0  or col==n_col-1) and from_query else getMaskB(points,channel,ids_xb,I[row,col])
                    axes[row, col].imshow(cropMask2(mask,crop_offset,value=255))
                axes[row, col].axis('off')
                
    else:
        row=0
        for col in range(n_col):
            if I[row,col] >=0:
                mask = getMaskQ(channel,ids_xb,I[row,col]) if (col==0  or col==n_col-1) and from_query else getMaskB(points,channel,ids_xb,I[row,col])
                axes[col].imshow(cropMask2(mask,crop_offset,value=255))
            axes[col].axis('off')
    
    plt.tight_layout()
    plt.show()


def visualizeTransformation(points,channel,ids_xb,SP_outputs,from_query=True, title = True, p=5):
    [series, distances] = SP_outputs
    n_row = 1
    n_col = len(series)
    
    
    fig, axes = plt.subplots(n_row, n_col,figsize=((n_col)*p,n_row*p))
    for col in range(n_col):
        mask = getMaskQ(channel,ids_xb,int(series[col])) if (col==0  or col==n_col-1) and from_query else getMaskB(points,channel,ids_xb,int(series[col]))
        if mask: axes[col].imshow(cropMask2(mask,offset=crop_offset,value=255))
        axes[col].axis('off')
    
    differences = [0]+list(np.diff(np.array(distances)))
    if title: [axes[col].set_title(f'Step {col}\nStep value: {values[0]:.2f}\nTotal distance: {values[1]:.2f}') for col,values in enumerate(zip(differences,distances))]
    
    plt.tight_layout()
    plt.show()



def visualizeQ(channel,idx_xq,idx,offset=crop_offset,
            title='',show_title=True ,axis='off',path_save=False):
    
    mask = getMaskQ(channel,idx_xq,idx)
    cropped = cropMask2(mask,offset)
    
    plt.figure()
    plt.imshow(cropped)
    if show_title: plt.title(title)
    plt.axis(axis)
    if path_save!=False: plt.savefig(join(path_save,f'{title}.jpg'),bbox_inches='tight')
    plt.show()


def visualizeB(points,channel,idx_xq,idx,offset=crop_offset,
            title='',show_title=True ,axis='off',path_save=False):
    
    mask = getMaskB(points,channel,idx_xq,idx)
    cropped = cropMask2(mask,offset)
    
    plt.figure()
    plt.imshow(cropped)
    if show_title: plt.title(title)
    plt.axis(axis)
    if path_save!=False: plt.savefig(join(path_save,f'{title}.jpg'),bbox_inches='tight')
    plt.show()