"""
Script that loads the AASIST model, evaluates it, generates 160-dimensional embeddings for train/dev/eval set, and saves it.

"""

# importing libraries
import argparse
import json
import os
import sys
import warnings
from importlib import import_module
from pathlib import Path
from shutil import copy
from typing import Dict, List, Union
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchcontrib.optim import SWA

from data_utils import (Dataset_ASVspoof2019_train,
                        Dataset_ASVspoof2019_devNeval, genSpoof_list)
from evaluation import calculate_tDCF_EER
from utils import create_optimizer, seed_worker, set_seed, str_to_bool

warnings.filterwarnings("ignore", category=FutureWarning)


# loading configuration file
config_path='config/AASIST.conf'

with open(config_path, "r") as f_json:
    config = json.loads(f_json.read())

optim_config = config["optim_config"]
optim_config["epochs"] = config["num_epochs"]
track = config["track"]
prefix_2019 = "ASVspoof2019.{}".format(track)

emb_config=config["embd_config"]

assert track in ["LA", "PA", "DF"], "Invalid track given"
if "eval_all_best" not in config:
    config["eval_all_best"] = "True"
if "freq_aug" not in config:
    config["freq_aug"] = "False"


# set file paths
database_path = Path(config["database_path"])
dev_trial_path = (database_path /
                      "ASVspoof2019_{}_cm_protocols/{}.cm.dev.trl.txt".format(
                          track, prefix_2019))
eval_trial_path = (
        database_path /
        "ASVspoof2019_{}_cm_protocols/{}.cm.eval.trl.txt".format(
            track, prefix_2019))
            

# defining data loaders
def get_loader(
        database_path: str,
        config: dict) -> List[torch.utils.data.DataLoader]:
    """Make PyTorch DataLoaders for train / developement / evaluation"""
    track = config["track"]
    prefix_2019 = "ASVspoof2019.{}".format(track)

    trn_database_path = database_path / "ASVspoof2019_{}_train/".format(track)
    dev_database_path = database_path / "ASVspoof2019_{}_dev/".format(track)
    eval_database_path = database_path / "ASVspoof2019_{}_eval/".format(track)

    trn_list_path = (database_path /
                     "ASVspoof2019_{}_cm_protocols/{}.cm.train.trn.txt".format(
                         track, prefix_2019))
    dev_trial_path = (database_path /
                      "ASVspoof2019_{}_cm_protocols/{}.cm.dev.trl.txt".format(
                          track, prefix_2019))
    eval_trial_path = (
        database_path /
        "ASVspoof2019_{}_cm_protocols/{}.cm.eval.trl.txt".format(
            track, prefix_2019))

    d_label_trn, file_train = genSpoof_list(dir_meta=trn_list_path,
                                            is_train=True,
                                            is_eval=False)
    print("no. training files:", len(file_train))

    # the modification is to use train set to obtain the embeddings
    train_set = Dataset_ASVspoof2019_devNeval(list_IDs=file_train,
                                           base_dir=trn_database_path)
    trn_loader = DataLoader(train_set,
                            batch_size=config["batch_size"],
                            shuffle=False,
                            drop_last=False,
                            pin_memory=True)

    _, file_dev = genSpoof_list(dir_meta=dev_trial_path,
                                is_train=False,
                                is_eval=False)
    print("no. validation files:", len(file_dev))

    dev_set = Dataset_ASVspoof2019_devNeval(list_IDs=file_dev,
                                            base_dir=dev_database_path)
    dev_loader = DataLoader(dev_set,
                            batch_size=config["batch_size"],
                            shuffle=False,
                            drop_last=False,
                            pin_memory=True)

    file_eval = genSpoof_list(dir_meta=eval_trial_path,
                              is_train=False,
                              is_eval=True)
    eval_set = Dataset_ASVspoof2019_devNeval(list_IDs=file_eval,
                                             base_dir=eval_database_path)
    eval_loader = DataLoader(eval_set,
                             batch_size=config["batch_size"],
                             shuffle=False,
                             drop_last=False,
                             pin_memory=True)

    return trn_loader, dev_loader, eval_loader


# data loaders for all
trn_loader, dev_loader, eval_loader = get_loader(
        database_path, config)


# loading the model
def get_model(model_config: Dict, device: torch.device):
    """define DNN model architecture"""
    module = import_module("models.{}".format(model_config["architecture"]))
    _model = getattr(module, "Model")
    model = _model(model_config).to(device)
    nb_params = sum([param.view(-1).size()[0] for param in model.parameters()])
    print("no. model params:{}".format(nb_params))

    return model

# set device
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device: {}".format(device))

# define model architecture
model_config = config["model_config"]
model = get_model(model_config, device)

model.load_state_dict(torch.load(config["model_path"], map_location=device))
print("Model loaded : {}".format(config["model_path"]))
print("Start evaluation...")

# checking the evaluation scores
def produce_evaluation_file(
    data_loader: DataLoader,
    model,
    device: torch.device,
    save_path: str,
    trial_path: str) -> None:
    """perform evaluation and save the score to a file"""
    model.eval()
    with open(trial_path, "r") as f_trl:
        trial_lines = f_trl.readlines()
    fname_list = []
    score_list = []
    for batch_x, utt_id in data_loader:
        batch_x = batch_x.to(device)
        with torch.no_grad():
            _, batch_out = model(batch_x)
            batch_score = (batch_out[:, 1]).data.cpu().numpy().ravel()
        # add outputs
        fname_list.extend(utt_id)
        score_list.extend(batch_score.tolist())

    assert len(trial_lines) == len(fname_list) == len(score_list)
    with open(save_path, "w") as fh:
        for fn, sco, trl in zip(fname_list, score_list, trial_lines):
            _, utt_id, _, src, key = trl.strip().split(' ')
            assert fn == utt_id
            fh.write("{} {} {} {}\n".format(utt_id, src, key, sco))
    print("Scores saved to {}".format(save_path))

# evaluation
eval_score_path=Path(emb_config["eval_score"])

produce_evaluation_file(eval_loader, model, device,
                                eval_score_path, eval_trial_path)

calculate_tDCF_EER(cm_scores_file=eval_score_path,
                           asv_score_file=database_path /
                           config["asv_score_path"],
                           output_file="t-DCF_EER.txt")


# extract embeddings
def generate_embeddings(
        data_loader: DataLoader,
        model,
        device: torch.device):
    
    model.eval()
    embs=torch.tensor([])
    for batch_x, utt_id in data_loader:
        batch_x=batch_x.to(device)
        with torch.no_grad():
            emb,_=model(batch_x)
            emb=emb.detach().data.cpu()
        embs=torch.cat((embs,emb))
    return embs.numpy()


# generating embeddings and saving in a folder
def save_embeddings(
        trn_loader, dev_loader, eval_loader,
        model,
        device: torch.device,
        emb_path):
    
    if not os.path.isdir(emb_path):
        os.mkdir(emb_path)
    
    embs_train=generate_embeddings(trn_loader,model,device)

    embs_dev=generate_embeddings(dev_loader,model,device)

    embs_eval=generate_embeddings(eval_loader,model,device)

    #save in files
    with open(emb_path/"train_emb.npy",'wb') as f:
        np.save(f, embs_train)

    with open(emb_path/"dev_emb.npy",'wb') as f:
        np.save(f, embs_dev)

    with open(emb_path/"eval_emb.npy",'wb') as f:
        np.save(f, embs_eval)

print("Start embedding extraction...")
save_embeddings(trn_loader,dev_loader,eval_loader,model,device,Path(emb_config["exp_dir"]))
print("Done")
