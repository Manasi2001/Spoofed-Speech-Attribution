import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as Data

class emb_fully_1(nn.Module):
    """
    A deep neural network architecture for spoken language recognition 
    """
    def __init__(self, idim, hdim, odim):
        super(emb_fully_1, self).__init__()
        self.layers = nn.ModuleList()
        cdim=idim
        #frame laval information capture using fc layers
        if hdim!= []:
            for dim in hdim:
                self.layers.append(nn.Linear(cdim, dim))
                cdim=dim
        self.layers.append(nn.Linear(cdim, odim))
    
        
    def forward(self,x):
        if len(self.layers[:-1]) > 0:
            for layer in self.layers[:-1]:
                x= F.relu(layer(x))
        x=self.layers[-1](x)
        return  x 
    

    
def initialize_weights(model_layer):
    if type(model_layer) == nn.Linear:
        torch.nn.init.xavier_normal_(model_layer.weight)
        #print(model_layer.weight)
    if type(model_layer) == nn.Conv1d:
        torch.nn.init.normal_(model_layer.weight, mean=0.0, std=0.9)
