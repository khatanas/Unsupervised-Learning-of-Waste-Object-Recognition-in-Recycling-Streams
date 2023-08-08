import torch
import importlib

def cudaInfo():
    torch.cuda.empty_cache()
    infos=torch.cuda.memory_summary()
    print(str(infos))
    
def reloadLib(lib_name:str):
    module = importlib.import_module(lib_name)
    importlib.reload(module)