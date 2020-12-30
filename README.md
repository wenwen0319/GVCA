# Generative-View-Correlation-Adaptation-for-Semi-Supervised-Multi-View-Learning

This is the implementatin of the ECCV'20 paper: [Generative-View-Correlation-Adaptation-for-Semi-Supervised-Multi-View-Learning](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123590307.pdf).

## Introduction

Multi-view learning (MVL) explores the data extracted from multiple resources. It assumes that the complementary information between different views could be revealed to further improve the learning performance. There are two challenges. First, it is difficult to effectively combine the different view data while still fully preserve the view-specific information. Second, multi-view datasets are usually small, which means the model can be easily overfitted. To address the challenges, we propose a novel View-Correlation Adaptation (VCA) framework in semisupervised fashion. A semi-supervised data augmentation me-thod is designed to generate extra features and labels based on both labeled and unlabeled samples. In addition, a cross-view adversarial training strategy is proposed to explore the structural information from one view and help the representation learning of the other view. Moreover, an effective and simple fusion network is proposed for the late fusion stage. In our model, all networks are jointly trained in an end-to-end fashion. Extensive experiments demonstrate that our approach is effective and stable compared with other state-of-the-art methods

## Implementation details

### The environment details:
* Ubuntu 16.04
* Python 3.5.5
* TensorFlow 1.5.0
* CUDA 9.0
* Cudnn 7

### File structure:
There are only one .py file as a demo for MUTAG dataset. The folder NUTAG_dataset contains the pre-arranged WEAVE encoding with different walk numbers per graph and walk length. Slight parameter modification is required to load different settings.

```
.
├── README.md                          
├── new_data                            
│     ├── DHA_total_test.csv
│     └── DHA_total_train.csv
├── loader_class_euc.py
└── GVCA.py
```

### Run the code
Simply run the python code:
```
python GVCA.py --d DHA
```
