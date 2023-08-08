import clip
import torch
import numpy as np
from os.path import join
from PIL import Image


def initCLIP(architecture='ViT-L/14@336px'):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load(architecture, device=device)
    return model,preprocess


def toNumpy(tensor):
    return tensor.detach().cpu().numpy().squeeze()


def getEmbedding(image_of_cropped_mask,model:tuple):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    image_clip =  model[1](Image.fromarray(image_of_cropped_mask)).unsqueeze(0).to(device)
    with torch.no_grad(): image_features = model[0].encode_image(image_clip)
    return toNumpy(image_features)



def computeScore(image_of_cropped_mask,list_of_categories, model:tuple):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    image_clip =  model[1](Image.fromarray(image_of_cropped_mask)).unsqueeze(0).to(device)
    text = clip.tokenize(list_of_categories).to(device)

    with torch.no_grad():
        logits_per_image,_ = model[0](image_clip, text)
        probs = logits_per_image.softmax(dim=-1).cpu().numpy()[0]
        
    return probs.max(), np.argmax(probs), probs


def getUk(embeddings, k):
    U,S,V = np.linalg.svd(embeddings)
    return U[:,:k]
    

def searchEmb(df,v1=0,v4=0,v7=0):
    final = df
    if 1 in df.columns:
        if v1 != 0:
            final = final[np.logical_and((v1-0.0001)<final[1],final[1]<(v1+0.0001))]
        if v4 != 0:
            final = final[np.logical_and((v4-0.0001)<final[4],final[4]<(v4+0.0001))]
        if v7 != 0:
            final = final[np.logical_and((v7-0.0001)<final[7],final[7]<(v7+0.0001))]
            
    elif '1' in df.columns:
        if v1 != 0:
            final = final[np.logical_and((v1-0.0001)<final['1'],final['1']<(v1+0.0001))]
        if v4 != 0:
            final = final[np.logical_and((v4-0.0001)<final['4'],final['4']<(v4+0.0001))]
        if v7 != 0:
            final = final[np.logical_and((v7-0.0001)<final['7'],final['7']<(v7+0.0001))]

    return final