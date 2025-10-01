Fork of the official PyTorch implementation of the paper "Exploiting Emotional Dependencies with Graph Convolutional Networks for Facial Expression Recognition" accepted at IEEE FG 2021.

The goal is to add support for audio/speech emotion recognition datasets to explore the viability of GCNs for multilingual SER.

Paper: [https://arxiv.org/abs/2106.03487](https://arxiv.org/abs/2106.03487)

Original authored by: Panagiotis Antoniadis, Panagiotis Paraskevas Filntisis, Petros Maragos

## Preparation

- Download the dataset. [[AffectNet]](http://mohammadmahoor.com/affectnet/) [[Aff-Wild2]](https://ibug.doc.ic.ac.uk/resources/aff-wild2/)
- Download the 300-dimensional GloVe vectors trained on the Wikipedia dataset from [here](https://drive.google.com/file/d/1d4A5LwOXTvtNBpzCTMNoUIc3TmlWs6R-/view?usp=sharing).
- Run `pickle_annotations_affectnet.py` and `pickle_annotations_affwild2.py` to generate `data_affectnet.pkl` and `data_affwild2.pkl`.

## Training

- Train Emotion-GCN on a FER dataset:

```
python main.py --help
usage: main.py [-h] [--image_dir IMAGE_DIR] [--data DATA] [--dataset {affectnet,affwild2}] [--network {densenet,bregnext}] [--adj ADJ]
               [--emb EMB] [--workers WORKERS] [--batch_size BATCH_SIZE] [--model {single_task,multi_task,emotion_gcn}] [--epochs EPOCHS]
               [--lambda_multi LAMBDA_MULTI] [--lr LR] [--momentum MOMENTUM] --gpu GPU --saved_model SAVED_MODEL

Train Facial Expression Recognition model using Emotion-GCN

optional arguments:
  -h, --help            show this help message and exit
  --image_dir IMAGE_DIR
                        path to images of the dataset
  --data DATA           path to the pickle file that holds all the information for each sample
  --dataset {affectnet,affwild2}
                        Dataset to use (default: affectnet)
  --network {densenet,bregnext}
                        Network to use (default: densenet)
  --adj ADJ             path to the pickle file that holds the adjacency matrix
  --emb EMB             path to the pickle file that holds the word embeddings
  --workers WORKERS     number of data loading workers (default: 4)
  --batch_size BATCH_SIZE
                        size of each batch (default: 35)
  --model {single_task,multi_task,emotion_gcn}
                        Model to use (default: emotion_gcn)
  --epochs EPOCHS       number of total epochs to train the network (default: 10)
  --lambda_multi LAMBDA_MULTI
                        lambda parameter of loss function
  --lr LR               learning rate (default: 0.001)
  --momentum MOMENTUM   momentum parameter of SGD (default: 0.9)
  --gpu GPU             id of gpu device to use
  --saved_model SAVED_MODEL
                        name of the saved model

```

## Pre-trained Models

We also provide weights for our Emotion-GCN models on AffectNet and Aff-Wild2. Our best model achieves 66.46% accuracy on the categorical model of AffectNet outperforming the performance of the current state-of-the-art methods for discrete FER. You can download the pre-trained models [here](https://drive.google.com/drive/folders/1BUUOKelxNtkIETrb93nb6VIP-J4bT7Os?usp=sharing).

## Citation
```
@inproceedings{9667014,
  author={Antoniadis, Panagiotis and Filntisis, Panagiotis Paraskevas and Maragos, Petros},
  booktitle={2021 16th IEEE International Conference on Automatic Face and Gesture Recognition (FG 2021)}, 
  title={Exploiting Emotional Dependencies with Graph Convolutional Networks for Facial Expression Recognition}, 
  year={2021},
  pages={1-8},
  doi={10.1109/FG52635.2021.9667014}}
```

## Contact
For questions feel free to open an issue.
