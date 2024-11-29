import gdown
import os
from .main import parse_args, MAIA2Model
from .utils import get_all_possible_moves, create_elo_dict
import torch
from torch import nn
import warnings
warnings.filterwarnings("ignore")

def from_pretrained(type, device, save_root = "./maia2_models"):
    
    if os.path.exists(save_root) == False:
        os.makedirs(save_root)
    
    if type == "blitz":
        url = "https://drive.google.com/uc?id=1X-Z4J3PX3MQFJoa8gRt3aL8CIH0PWoyt"
        output_path = os.path.join(save_root, "blitz_model.pt")
    
    elif type == "rapid":
        url = "https://drive.google.com/uc?id=1gbC1-c7c0EQOPPAVpGWubezeEW8grVwc"
        output_path = os.path.join(save_root, "rapid_model.pt")
    
    else:
        raise ValueError("Invalid model type. Choose between 'blitz' and 'rapid'.")

    if os.path.exists(output_path):
        print(f"Model for {type} games already downloaded.")
    else:
        print(f"Downloading model for {type} games.")
        gdown.download(url, output_path, quiet=False)

    cfg = parse_args()

    all_moves = get_all_possible_moves()
    elo_dict = create_elo_dict()

    model = MAIA2Model(len(all_moves), elo_dict, cfg)
    model = nn.DataParallel(model)
    
    checkpoint = torch.load(output_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.module
    
    if device == "gpu":
        model = model.cuda()
    
    print(f"Model for {type} games loaded to {device}.")
    
    return model
    
    
    
    
    
    
