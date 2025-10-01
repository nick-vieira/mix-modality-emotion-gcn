import os
import cv2
import torch
import pickle
import numpy as np
import pandas as pd
import re
import librosa
import matplotlib.pyplot as plt
import matplotlib.style as ms
from torch.nn.utils.rnn import pad_sequence
from joblib import Parallel, delayed
from PIL import Image
from torch.utils.data import Dataset
from utils import extract_acoustic_vector

class AffectNet_annotation(object):
    """A class that represents a sample from AffectNet."""

    def __init__(self, image_path, face_x, face_y, face_width, face_height, expression, valence, arousal, left_eye, right_eye):
        super(AffectNet_annotation, self).__init__()
        self.image_path = image_path
        self.face_x = face_x
        self.face_y = face_y
        self.face_width = face_width
        self.face_height = face_height
        self.expression = expression
        self.valence = valence
        self.arousal = arousal
        self.left_eye = left_eye
        self.right_eye = right_eye

class AffectNet_dataset(Dataset):
    """AffectNet: Facial expression recognition dataset."""

    def __init__(self, root_dir, data_pkl, emb_pkl, aligner, train=True, transform=None, crop_face=True):
        self.root_dir = root_dir
        self.aligner = aligner
        self.train = train
        self.transform = transform
        self.crop_face = crop_face

        data_pickle = pickle.load(open(data_pkl, 'rb'))
        if train:
            self.data = data_pickle['train']
        else:
            self.data = data_pickle['val']

        self.inp = torch.load(emb_pkl)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        # Get selected sample
        sample = self.data[idx]
        # Read image
        img_name = os.path.join(self.root_dir, sample.image_path)
        image = cv2.imread(img_name)[..., ::-1]

        image, M = self.aligner.align(image, sample.left_eye, sample.right_eye)
        if self.crop_face:
            # Keep only the face region of the image
            face_x = int(sample.face_x)
            face_y = int(sample.face_y)
            face_width = int(sample.face_width)
            face_height = int(sample.face_height)

            point = (face_x, face_y)
            rotated_point = M.dot(np.array(point + (1,)))
            image = image.crop(
                (rotated_point[0], rotated_point[1], rotated_point[0]+face_width, rotated_point[1]+face_height))

        # Read the expression of the image
        expression = sample.expression
        # Apply the transformation
        if self.transform:
            image = self.transform(image)

        cont = np.array([sample.valence, sample.arousal])

        return image, expression, cont, self.inp


class Affwild2_annotation(object):
    """A class that represents a sample from Aff-Wild2."""

    def __init__(self, frame_path, expression, valence, arousal):
        super(Affwild2_annotation, self).__init__()
        self.frame_path = frame_path
        self.expression = expression
        self.valence = valence
        self.arousal = arousal

class Affwild2_dataset(Dataset):
    """Aff-Wild2"""

    def __init__(self, data_pkl, emb_pkl, train=True, transform=None):

        self.train = train
        self.transform = transform

        data_pickle = pickle.load(open(data_pkl, 'rb'))
        if train:
            self.data = data_pickle['train']
        else:
            self.data = data_pickle['val']

        self.inp = torch.load(emb_pkl)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        sample = self.data[idx]
        img_name = sample.frame_path
        image = Image.open(img_name).convert("RGB")

        expression = sample.expression
        if self.transform:
            image = self.transform(image)

        cont  = np.array([sample.valence, sample.arousal])
        
        return image, expression, cont, self.inp
                
class IEMOCAP_dataset(Dataset):
    """
    IEMOCAP audio dataset object specifically designed for interfacing with the GCN
    Returns:
        - graph: torch.FloatTensor (N, N) adjacency matrix (same for all samples)
        - feats: torch.FloatTensor (N, F) acoustic node features
        - label: int (emotion class)
    """
    def __init__(self, train=True, adj_file='./adj.pkl'):
        self.videoIDs, self.videoSpeaker, self.videoLabels, self.videoText, \
        self.videoAudio, self.videoVisual, self.videoSentence, self.trainVid,\
        self.testVid = pickle.load(open('./iemocap_features/IEMOCAP_features.pkl', 'rb'), encoding='utf8')

        self.keys = [x for x in (self.trainVid if train else self.testVid)]
        self.len = len(self.keys)
        self.adj = torch.load(adj_file)
        self.labels = self.videoLabels

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        vid=self.keys[index]
        wav_path = self.videoAudio[vid]
        acoustic_vec = extract_acoustic_vector(wav_path=wav_path)
        N = self.adj.size(0)
        node_feature = acoustic_vec.unsqueeze(0).repeat(N,1)
        label = int(self.labels[vid])
        return torch.FloatTensor(self.adj.clone()), node_feature, label
        # return torch.FloatTensor(self.videoText[vid]),\
        #     torch.FloatTensor(self.videoVisual[vid]),\
        #     torch.FloatTensor(self.videoAudio[vid]),\
        #     torch.FloatTensor([[1,0] if x=='M' else [0,1] for x in\
        #                         self.videoSpeakers[vid]]),\
        #     torch.FloatTensor([1]*len(self.videoLabels[vid])),\
        #     torch.LongTensor(self.videoLabels[vid]),\
        #     vid