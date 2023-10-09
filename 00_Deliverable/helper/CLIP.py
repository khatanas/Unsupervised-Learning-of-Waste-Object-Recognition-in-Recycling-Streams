from config_.parameters import arch_CLIP
from helper.common_libraries import torch,np,Image
import clip

#**************************************************************************************************************
# creates text embeddings for the provided list_of_categories
tokenizeText = lambda list_of_categories: clip.tokenize(list_of_categories).to("cuda" if torch.cuda.is_available() else "cpu")

# convert the CLIP output tensor to a numpy array
toNumpy = lambda tensor: tensor.detach().cpu().numpy().squeeze()

def initCLIP(architecture=arch_CLIP):
    """
    Initializes CLIP with architecture {arch_CLIP}
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load(architecture, device=device)
    return model,preprocess


def getEmbedding(image,model:tuple):
    """
    Returns the {image} embeddings
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    image_clip =  model[1](Image.fromarray(image)).unsqueeze(0).to(device)
    with torch.no_grad(): image_features = model[0].encode_image(image_clip)
    return toNumpy(image_features)


def computeScore(image, tokenized_categories, model:tuple):
    """
    Assigns a category to an image
    Returns the the best probability score, its position in the list, and the list of all computed scores
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    image_clip =  model[1](Image.fromarray(image)).unsqueeze(0).to(device)
    
    with torch.no_grad():
        logits_per_image,_ = model[0](image_clip, tokenized_categories)
        probs = logits_per_image.softmax(dim=-1).cpu().numpy()[0]
        
    return float(probs.max()), int(np.argmax(probs)), probs