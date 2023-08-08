from config.parameters import arch_CLIP

from helper.common_libraries import clip,torch,np,Image


def initCLIP(architecture=arch_CLIP):
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
        
    return float(probs.max()), int(np.argmax(probs)), probs