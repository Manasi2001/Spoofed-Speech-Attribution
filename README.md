# Spoofed Speech Attribution

This repository focuses on extending the functionality of the ['AASIST: Audio Anti-Spoofing using Integrated Spectro-Temporal Graph Attention Networks'](https://arxiv.org/abs/2110.01200)
 model to predict attributes that characterize spoofed speech. The approach introduces a bank of probabilistic detectors that are trained to identify specific features associated with selected spoofing techniques. This results in a comprehensive attribute-based representation of each audio sample. This representation is then analyzed using decision tree modeling to enable accurate spoofed speech detection and detailed explanations for the model's decisions. The dataset selected for the experiments is LA scenario of ASVSpoof 2019.

![full_arch](https://github.com/Manasi2001/Spoofed-Speech-Attribution/assets/68627617/1478fe33-27e8-4814-8c3e-09cceed162cf)

**Figure:** Complete implementation workflow of the proposed architecture for explainable spoofed speech detection. **Phase I** demonstrates the extraction of embeddings using the AASIST model and the subsequent processing of these embeddings through a bank of seven probabilistic feature detectors. **Phase II** illustrates the concatenation of the outputs from these detectors to create a 25-unit long vector, which is then fed into a decision tree model for classification. This decision tree model is used for both bonafide/spoofed classification and spoofing attack algorithm characterization.

### Getting started

`requirements.txt` must be installed for execution. 

```
pip install -r requirements.txt
```

### Data preparation

To download the ASVspoof 2019 logical access dataset [2]:

```
python download_dataset.py
```

(Alternative) Manual preparation is available via 
- ASVspoof2019 dataset: https://datashare.ed.ac.uk/handle/10283/3336
  1. Download `LA.zip` and unzip it.
  2. Set the dataset directory in the configuration file.
