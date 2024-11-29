import gdown
import os
import pandas as pd

def load_example_test_dataset(save_root = "./maia2_data"):
    
    url = "https://drive.google.com/uc?id=1fSu4Yp8uYj7xocbHAbjBP6DthsgiJy9X"
    if os.path.exists(save_root) == False:
        os.makedirs(save_root)
    output_path = os.path.join(save_root, "example_test_dataset.csv")
    
    if os.path.exists(output_path):
        print("Example test dataset already downloaded.")
    else:
        gdown.download(url, output_path, quiet=False)
        print("Example test dataset downloaded.")
        
    data = pd.read_csv(output_path)
    data = data[data.move_ply > 10][['board', 'move', 'active_elo', 'opponent_elo']]
    
    return data
    
def load_example_train_dataset(save_root = "./maia2_data"):
    
    url = "https://drive.google.com/uc?id=1XBeuhB17z50mFK4tDvPG9rQRbxLSzNqB"
    if os.path.exists(save_root) == False:
        os.makedirs(save_root)
    output_path = os.path.join(save_root, "example_train_dataset.csv")
    
    if os.path.exists(output_path):
        print("Example train dataset already downloaded.")
    else:
        gdown.download(url, output_path, quiet=False)
        print("Example train dataset downloaded.")
    
    return output_path
    