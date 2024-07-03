import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
np.float=float
def create_training_labels(attribute):

    # data based on "Table 1: Summary of LA spoofing systems." in "ASVspoof 2019: a large-scale public database of synthetized, converted and replayed speech"
    data = {"A01": [1,0,1,0,0,1,0,0,1,0,0,0,0,1,0,0,1,0,0,0,0,1,0,0,0],
            "A02": [1,0,1,0,0,1,0,0,1,0,0,0,0,1,0,0,0,1,0,0,0,0,1,0,0],
            "A03": [1,0,1,0,0,0,1,0,0,1,0,0,0,0,1,0,0,1,0,0,0,0,1,0,0],
            "A04": [1,0,1,0,0,0,0,1,0,0,1,0,0,0,0,1,0,0,1,0,0,0,0,1,0],
            "A05": [0,1,0,1,0,0,0,1,0,0,0,1,0,0,1,0,0,0,0,1,0,0,1,0,0],
            "A06": [0,1,0,0,1,0,0,1,0,0,0,0,1,0,0,1,0,0,0,0,1,0,0,0,1],

            "A07": [1,0,1,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0],
            "A08": [1,0,1,0,0,1,0,0,1,0,0,0,0,0,1,0,1,0,0,0,0,0,0,0,0],
            "A09": [1,0,1,0,0,0,0,0,0,0,0,0,0,0,1,0,1,0,0,0,0,0,0,0,0],
            "A10": [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0], 
            "A11": [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
            "A12": [1,0,1,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0],
            "A13": [0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0],
            "A14": [0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,1,0,0,0,0,0,0,0],
            "A15": [0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,1,0,0,0,0,1,0,0,0],
            "A16": [1,0,1,0,0,0,0,1,0,0,1,0,0,0,0,1,0,0,1,0,0,0,0,1,0],
            "A17": [0,1,0,1,0,0,0,1,0,0,0,1,0,0,1,0,1,0,0,0,0,0,0,0,0],
            "A18": [0,1,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
            "A19": [0,1,0,0,1,0,0,1,0,0,0,0,1,0,0,1,0,0,0,0,1,0,0,0,1]} 
    
    return data[attribute]


#embedding = create_training_labels('A19')
#print(embedding)
#%%
def spoof_info(data_path, txt_path):
    data=np.load(data_path)
    
    df = pd.read_csv(txt_path, sep=" ", header=None)
    s_b=df[4].values
    idx_spoof=np.array([x.strip()=='spoof' for x in s_b])

    # take only spoofed targets
    data=data[idx_spoof,:]
    data_conf=df[3].values[idx_spoof]
    data_list=df[1].values[idx_spoof]
    return data, data_conf, data_list

#%%
def all_info(data_path):
    data=np.load(data_path)
    return data

#%%
class Dataloader_emb(Dataset):
    def __init__(self, data, data_conf, create_training_labels):
        """self.list_IDs	: list of strings (each string: utt key),
        """
        self.emb= torch.tensor(data)
        self.conf=data_conf
        self.lab=create_training_labels

    def __len__(self):
        return len(self.conf)

    def __getitem__(self, index):

        x=self.emb[index,:]
        y=torch.tensor(self.lab(self.conf[index]))

        return x,y
#%%
class Dataloader_emb_all(Dataset):
    def __init__(self, data):
        """self.list_IDs	: list of strings (each string: utt key),
        """
        self.emb= torch.tensor(data)

    def __len__(self):
        return len(self.emb)

    def __getitem__(self, index):

        x=self.emb[index,:]
        return x