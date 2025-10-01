import torch
import numpy as np, argparse, time, pickle, random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataloading import IEMOCAP_dataset
from torch.utils.data.sampler import SubsetRandomSampler
from models import Emotion_GCN, multi_densenet, BReGNeXt, BReGNeXt_GCN
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score, classification_report, precision_recall_fscore_support

def train_model_single(dataloader, model, criterion, optimizer):
    model.train()
    running_loss = 0.0
    y_pred = []
    y_true = []

    device = next(model.parameters()).device
    for index, batch in enumerate(dataloader, 1):
        inputs, labels, _, _ = batch

        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)

        loss = criterion(outputs, labels)
        _, preds = torch.max(outputs, 1)

        loss.backward()
        optimizer.step()

        running_loss += loss.data.item()

        y_pred.append(preds.cpu().numpy())
        y_true.append(labels.cpu().numpy())

    return running_loss / index, (y_true, y_pred)


def eval_model_single(dataloader, model, criterion):
    model.eval()
    running_loss = 0.0
    y_pred = []
    y_true = []

    device = next(model.parameters()).device
    with torch.no_grad():
        for index, batch in enumerate(dataloader, 1):
            inputs, labels, _, _ = batch

            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)

            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)

            y_pred.append(preds.cpu().numpy())
            y_true.append(labels.cpu().numpy())

            running_loss += loss.data.item()

    return running_loss / index, (y_true, y_pred)
    
def train_model_multi(dataloader, model, criterion_cat, criterion_cont, optimizer, gcn=False):
    model.train()
    running_loss = 0.0
    running_loss_cat = 0.0
    running_loss_cont = 0.0
    y_pred = []
    y_true = []

    device = next(model.parameters()).device
    for index, batch in enumerate(dataloader, 1):
        inputs, labels, labels_cont, inp = batch

        inputs = inputs.to(device)
        labels = labels.to(device)
        labels_cont = labels_cont.to(device)
        inp = inp.to(device)

        optimizer.zero_grad()

        if gcn:
            outputs_cat, outputs_cont = model(inputs, inp)
        else:
            outputs_cat, outputs_cont = model(inputs)

        loss_cat = criterion_cat(outputs_cat, labels)
        loss_cont = criterion_cont(outputs_cont.double(), labels_cont.double())
        _, preds = torch.max(outputs_cat, 1)
        loss = loss_cat + loss_cont

        loss.backward()
        optimizer.step()

        running_loss += loss.data.item()
        running_loss_cat += loss_cat.data.item()
        running_loss_cont += loss_cont.data.item()

        y_pred.append(preds.cpu().numpy())
        y_true.append(labels.cpu().numpy())

    return running_loss / index, running_loss_cat / index, running_loss_cont / index, (y_true, y_pred)


def eval_model_multi(dataloader, model, criterion_cat, criterion_cont, gcn=False):
    model.eval()
    running_loss = 0.0
    running_loss_cat = 0.0
    running_loss_cont = 0.0
    y_pred = []
    y_true = []

    device = next(model.parameters()).device
    with torch.no_grad():
        for index, batch in enumerate(dataloader, 1):
            inputs, labels, labels_cont, inp = batch

            inputs = inputs.to(device)
            labels = labels.to(device)
            labels_cont = labels_cont.to(device)
            inp = inp.to(device)

            if gcn:
                outputs_cat, outputs_cont = model(inputs, inp)
            else:
                outputs_cat, outputs_cont = model(inputs)

            loss_cat = criterion_cat(outputs_cat, labels)
            loss_cont = criterion_cont(
                outputs_cont.double(), labels_cont.double())
            _, preds = torch.max(outputs_cat, 1)
            loss = loss_cat + loss_cont

            y_pred.append(preds.cpu().numpy())
            y_true.append(labels.cpu().numpy())

            running_loss += loss.data.item()
            running_loss_cat += loss_cat.data.item()
            running_loss_cont += loss_cont.data.item()

    return running_loss / index, running_loss_cat / index, running_loss_cont / index, (y_true, y_pred)

def get_train_valid_sampler(trainset, valid=0.1):
    size = len(trainset)
    idx = list(range(size))
    split = int(valid*size)
    return SubsetRandomSampler(idx[split:]), SubsetRandomSampler(idx[:split])

def get_IEMOCAP_loaders(batch_size = 32, valid=0.1, num_workers=4, pin_memory=False):
    train_dataset = IEMOCAP_dataset()
    train_sampler, valid_sampler = get_train_valid_sampler(train_dataset, valid)
    test_dataset = IEMOCAP_dataset(train=False)

    train_loader = DataLoader(train_dataset,
                              batch_size=batch_size,
                              sampler=train_sampler,
                              collate_fn=train_dataset.collate_audio_fn,
                              num_workers=num_workers,
                              pin_memory=pin_memory)

    valid_loader = DataLoader(train_dataset,
                              batch_size=batch_size,
                              sampler=valid_sampler,
                              collate_fn=train_dataset.collate_audio_fn,
                              num_workers=num_workers,
                              pin_memory=pin_memory)        

    test_loader = DataLoader(train_dataset,
                              batch_size=batch_size,
                              sampler=train_sampler,
                              collate_fn=test_dataset.collate_audio_fn,
                              num_workers=num_workers,
                              pin_memory=pin_memory)
    
    return train_loader, valid_loader, test_loader
