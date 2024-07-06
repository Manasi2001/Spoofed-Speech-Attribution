"""
Script for designing probabilistic detectors for different attribute sets using user-specified architectures.

"""

import json
import os
import numpy as np
import warnings
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch import optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
from scipy.special import logit

warnings.filterwarnings("ignore", category=FutureWarning)

from fetch_data_function import Dataloader_emb,spoof_info,create_training_labels
from emb_model import emb_fully_1, initialize_weights
from evaluation import compute_eer
from utils import create_optimizer

import configparser

config = configparser.ConfigParser()
config.read('emb_model_AASIST.conf')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

log_dir = "probabilistic_detectors/" + 'AS' + config['attribute']['type'] + '_' + config['model-config']['hdim'] 
os.makedirs(log_dir, exist_ok=True)

# Function to save model
def save_model(model, filename):
    torch.save(model.state_dict(), os.path.join(log_dir, filename))

# Function to log statistics
def log_stats(stats):
    with open(os.path.join(log_dir, 'log.txt'), 'a') as f:
        f.write(json.dumps(stats) + '\n')

def attribut_type_index(att_type):
    if att_type==1:
        st=0
        en=2
    if att_type==2:
        st=2
        en=5
    if att_type==3:
        st=5
        en=8
    if att_type==4:
        st=8
        en=13
    if att_type==5:
        st=13
        en=16
    if att_type==6:
        st=16
        en=21
    if att_type==7:
        st=21
        en=25
    return st,en

st, en = attribut_type_index(int(config['attribute']['type']))

print(f"Attribute type [{int(config['attribute']['type'])}], Start: {st}, End: {en}")

# Function to calculate accuracy
def calculate_accuracy(outputs, targets):
    outputs = F.softmax(outputs, dim=1)
    outputs_args = torch.argmax(outputs, dim = 1)
    targets_args = torch.argmax(targets, dim = 1)
    
    acc = torch.sum(outputs_args == targets_args).item() / len(targets)
    
    return acc

# Function to calculate L1 distance
def calculate_l1(outputs, targets):
    return torch.mean(torch.abs(outputs - targets)).item()

# Function to compute equal error rate
def compute_EER(output, target):

    true_score=torch.tensor([])
    false_score=torch.tensor([])
    attributes_eer = []
    output=logit(F.softmax(output, dim=1))
    for i in range(output.shape[1]):
        ts=output[target[:,i]==1,i]
        fs=output[target[:,i]!=1,i]
        true_score=torch.cat((true_score,ts))
        false_score=torch.cat((false_score,fs))
        eer = compute_eer(true_score.numpy(), false_score.numpy())[0]
        eer_indi = compute_eer(ts.numpy(), fs.numpy())[0]
        attributes_eer.append(eer_indi*100)

    return eer, attributes_eer

# Function to compute weights in the training set
def compute_weights(train_data):

    tmp = DataLoader(train_data, batch_size=len(train_data), shuffle=False, drop_last=False, pin_memory=True)

    data = next(iter(tmp))
    target = data[1][:, st:en]
    target = target.numpy()
    flattened_targets = np.argmax(target, axis=1)
    unq, counts = np.unique(flattened_targets, return_counts=True)
    class_weights = flattened_targets.shape[0]/(unq.shape[0]*counts)
        #class_weights = dict(zip(np.unique(target), class_weights))
    print(class_weights)  
    return class_weights

# Main function
def main():
    # Define data loaders
    dev_data1, dev_conf,_ = spoof_info(config['emb-path']['dev_emb'], config['data-path']['dev_data'])
    dev_data = Dataloader_emb(dev_data1, dev_conf, create_training_labels)
    dev_loader = DataLoader(dev_data, batch_size = int(config['model-config']['batch_size']), shuffle=False, drop_last=False, pin_memory=True)

    train_data1, train_conf,_ = spoof_info(config['emb-path']['train_emb'], config['data-path']['train_data'])
    train_data = Dataloader_emb(train_data1, train_conf,create_training_labels)
    train_loader = DataLoader(train_data, batch_size = int(config['model-config']['batch_size']), shuffle=True, drop_last=False, pin_memory=True)

    eval_data1,eval_conf,_ = spoof_info(config['emb-path']['eval_emb'], config['data-path']['eval_data'])
    eval_data = Dataloader_emb(eval_data1, eval_conf,create_training_labels)
    eval_loader = DataLoader(eval_data, batch_size = int(config['model-config']['batch_size']), shuffle=False, drop_last=False, pin_memory=True)

    # Define model
    in_dim = int(config['model-config']['in_dim'])
    out_dim = int(config['model-config']['out_dim'])
    hdim = json.loads(config['model-config']['hdim'])
    model = emb_fully_1(idim=in_dim, hdim=hdim, odim=out_dim).to(device)
    model_arch = model
    with open(os.path.join(log_dir, 'log.txt'), 'w') as f:
        print(model_arch, file=f)
        print('\n\n', file=f)
        config.write(f)
        print('\n\n', file=f)
    model.apply(initialize_weights)

    # Define loss function and optimizer
    weights_dict = compute_weights(train_data)
    #weight_0 = weights_dict[0.0]
    #weight_1 = weights_dict[1.0]
    weight = torch.FloatTensor(weights_dict).to(device)
    #print(weight)
    class_obj = nn.CrossEntropyLoss(weight=weight)
    lr = float(config['model-config']['lr'])
    total_epochs = int(config['model-config']['total_epochs'])
    optimizer = optim.Adam(model.parameters(), lr=lr)
    # get optimizer and scheduler
    #optim_config["steps_per_epoch"] = len(train_loader)
    #optimizer, scheduler = create_optimizer(model.parameters(), optim_config)

    # Training loop
    training_epc_loss = []
    dev_epc_loss = []
    best_train_loss = float('inf')
    best_dev_loss = float('inf')
    best_eer = float('inf')
    best_l1_distance = float('inf')
    best_accuracy = float('-inf')

    train_loss_npy = os.path.join(log_dir, 'train_loss.npy')
    dev_loss_npy = os.path.join(log_dir, 'dev_loss.npy')
    accuracy_npy = os.path.join(log_dir, 'accuracy.npy')
    eer_npy = os.path.join(log_dir, 'eer.npy')
    l1_distance_npy = os.path.join(log_dir, 'l1_distance.npy')

    for epoch in range(total_epochs):
        run_loss = 0.
        model.train()

        for id, data in enumerate(train_loader):
            optimizer.zero_grad()

            emb_vecs = data[0].to(dtype=torch.float).to(device)
            target = data[1][:, st:en].to(dtype=torch.float).to(device)

            output = model(emb_vecs)
            loss = class_obj(output, target)
            loss.backward()
            optimizer.step()
            run_loss += loss.item()
            '''
            if config["optim_config"]["scheduler"] in ["cosine", "keras_decay"]:
                scheduler.step()
            elif scheduler is None:
                pass
            else:
                raise ValueError("scheduler error, got:{}".format(scheduler))

            '''
        avg_loss = run_loss / len(train_loader)
        training_epc_loss.append(avg_loss)

        # Validation
        model.eval()
        with torch.no_grad():
            dev_loss = 0.
            all_preds = []
            all_targets = []
            for data in dev_loader:
                emb_vecs = data[0].to(dtype=torch.float).to(device)
                # extra
                # print(data[2][:, :2])
                target = data[1][:, st:en].to(dtype=torch.float).to(device)

                output = model(emb_vecs)
                dev_loss += class_obj(output, target).item()

                all_preds.extend(output.cpu().numpy())
                all_targets.extend(target.cpu().numpy())

            dev_loss /= len(dev_loader)
            dev_epc_loss.append(dev_loss)
            acc = calculate_accuracy(torch.tensor(all_preds), torch.tensor(all_targets))
            l1_distance = calculate_l1(torch.tensor(all_preds), torch.tensor(all_targets))
            eer, _ = compute_EER(torch.tensor(all_preds), torch.tensor(all_targets))

            # Save best model based on dev loss
            if dev_loss < best_dev_loss:
                best_dev_loss = dev_loss
                best_dev_loss_epoch =epoch+1
                save_model(model, 'best_dev_loss_model.pth')
            if avg_loss < best_train_loss:
                best_train_loss = avg_loss
                best_train_loss_epoch = epoch+1
                save_model(model, 'best_train_loss_model.pth')
            if eer < best_eer:
                best_eer = eer
                best_eer_epoch=epoch+1
                save_model(model, 'best_eer_model.pth')
            if l1_distance < best_l1_distance:
                best_l1_distance = l1_distance
                best_l1_epoch=epoch+1
                save_model(model, 'best_l1_distance_model.pth')
            if acc > best_accuracy:
                best_accuracy = acc
                best_accuracy_epoch=epoch+1
                save_model(model, 'best_accuracy_model.pth')

            with open(train_loss_npy, 'ab') as f:  
                np.savetxt(f, [avg_loss], delimiter=',')
            with open(dev_loss_npy, 'ab') as f:  
                np.savetxt(f, [dev_loss], delimiter=',')
            with open(eer_npy, 'ab') as f:  
                np.savetxt(f, [eer], delimiter=',')
            with open(l1_distance_npy, 'ab') as f:  
                np.savetxt(f, [l1_distance], delimiter=',')
            with open(accuracy_npy, 'ab') as f:  
                np.savetxt(f, [acc], delimiter=',')

            # Log statistics
            stats = {
                'epoch': epoch + 1,
                'train_loss': avg_loss,
                'dev_loss': dev_loss,
                'accuracy': acc,
                'eer': best_eer,
                'l1_distance': l1_distance
                # Include more statistics as needed
            }
            log_stats(stats)

            # Accuracy: {acc:.4f}, 

            print(f"Epoch [{epoch + 1}/{total_epochs}], Train Loss: {avg_loss:.4f}, Dev Loss: {dev_loss:.4f}, Accuracy: {acc:.4f}, EER: {best_eer:.4f}, L1 Distance: {l1_distance:.4f}")

    with open(os.path.join(log_dir, 'log.txt'), 'a') as f:
        print('\n\n', file=f)
        print('Best Train Loss: ' + str(best_train_loss)+' at epoch: '+str(best_train_loss_epoch), file=f)
        print('\n', file=f)
        print('Best Dev Loss: ' + str(best_dev_loss)+' at epoch: '+str(best_dev_loss_epoch), file=f)
        print('\n', file=f)
        print('Best Accuracy: ' + str(best_accuracy)+' at epoch: '+str(best_accuracy_epoch), file=f)
        print('\n', file=f)
        print('Best EER: ' + str(best_eer)+' at epoch: '+str(best_eer_epoch), file=f)
        print('\n', file=f)
        print('Best L1 Distance: ' + str(best_l1_distance)+' at epoch: '+str(best_l1_epoch), file=f)

        # Loss vs Epoch array
        plt.plot(training_epc_loss)
        plt.xlabel('Epoch')
        plt.ylabel('Cross Entropy Loss')
        plt.title('Training Loss vs Epoch')
        plt.grid(True)
        plt.savefig(os.path.join(log_dir, 'training_loss_vs_epoch.png'))
        plt.close()

        plt.plot(dev_epc_loss)
        plt.xlabel('Epoch')
        plt.ylabel('Cross Entropy Loss')
        plt.title('Development Loss vs Epoch')
        plt.grid(True)
        plt.savefig(os.path.join(log_dir, 'dev_loss_vs_epoch.png'))
        plt.close()

if __name__ == "__main__":
    main()
