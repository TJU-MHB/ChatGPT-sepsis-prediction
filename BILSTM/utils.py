import os
import time
import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset, Dataset
import torchmetrics
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, matthews_corrcoef,auc,precision_recall_curve,confusion_matrix,roc_curve
import seaborn as sns
import matplotlib.pyplot as plt
import torch.nn as nn


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def compute_batch_rocauc(output, target):
    """Computes the ROC AUC score for a batch"""
    with torch.no_grad():
        batch_size = target.size(0)

        # Apply softmax to output to get predicted probabilities
        probs = torch.softmax(output, dim=1)

        # Compute ROC AUC score
        rocauc = torchmetrics.functional.auroc(probs[:, 1], target, task='binary')

        return rocauc * 100.0
def bootstrap_confidence_interval(data, n_bootstrap, confidence_level):

    bootstrap_samples = np.empty((n_bootstrap, len(data)))
    for i in range(n_bootstrap):
        bootstrap_sample = np.random.choice(data, size=len(data), replace=True)
        bootstrap_samples[i] = bootstrap_sample
    bootstrap_means = np.mean(bootstrap_samples, axis=1)
    lower_percentile = (1 - confidence_level) / 2
    upper_percentile = 1 - lower_percentile
    confidence_interval = np.percentile(bootstrap_means, [lower_percentile * 100, upper_percentile * 100])

    return confidence_interval


def calculate_num_features(seqs):
    """
    :param seqs:
    :return: the calculated number of features
    """
    return len(seqs[0][0])


class VisitSequenceWithLabelDataset(Dataset):
    def __init__(self, seqs, labels):
        """
        Args:
            seqs (list): list of patients (list) of visits (list) of codes (int) that contains visit sequences
            labels (list): list of labels (int)
        """
        self.labels = labels
        matrix_seqs = []
        for record_seqs in seqs:
            matrix_seqs.append(np.matrix(record_seqs))
        self.seqs = matrix_seqs

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return self.seqs[index], self.labels[index]


def time_collate_fn(batch):
    """
    DataLoaderIter call - self.collate_fn([self.dataset[i] for i in indices])
    Thus, 'batch' is a list [(seq1, label1), (seq2, label2), ... , (seqN, labelN)]
    where N is minibatch size, seq is a FloatTensor, and label is a LongTensor

    :returns
        seqs (FloatTensor) - 3D of batch_size X max_length X num_features
        lengths (LongTensor) - 1D of batch_size
        labels (LongTensor) - 1D of batch_size
    """
    batch_seq, batch_label = zip(*batch)

    num_features = batch_seq[0].shape[1]
    seq_lengths = list(map(lambda patient_tensor: patient_tensor.shape[0], batch_seq))
    max_length = 7

    sorted_indices, sorted_lengths = zip(*sorted(enumerate(seq_lengths), key=lambda x: x[1], reverse=True))
    sorted_padded_seqs = []
    sorted_labels = []

    for i in sorted_indices:
        length = batch_seq[i].shape[0]

        if length < max_length:
            padded_patient = np.concatenate((batch_seq[i].tolist(),
                                             np.zeros((max_length - length, num_features))), axis=0)
        else:
            padded_patient = batch_seq[i].tolist()

        sorted_padded_seqs.append(padded_patient)
        sorted_labels.append(batch_label[i])

    seqs_tensor = torch.cuda.FloatTensor(np.stack(sorted_padded_seqs, axis=0))
    lengths_tensor = torch.cuda.LongTensor(list(sorted_lengths))
    labels_tensor = torch.cuda.LongTensor(sorted_labels)

    return (seqs_tensor, lengths_tensor), labels_tensor

def train(model, device, data_loader, criterion, optimizer, epoch):

    losses = AverageMeter()
    model.train()
    results = []
    for i, (input, target) in enumerate(data_loader):
        if isinstance(input, tuple):
            input = tuple([e.to(device) if type(e) == torch.Tensor else e for e in input])
        else:
            input = input.to(device)
        target = target.to(device)

        optimizer.zero_grad()
        output = model(input)
        loss = criterion(output, target)
        assert not np.isnan(loss.item())

        loss.backward()
        optimizer.step()
        losses.update(loss.item(), target.size(0))

        y_true = target.detach().to('cpu').numpy().tolist()
        y_pred = nn.Softmax(1)(output).detach().to('cpu').numpy()[:,1].tolist()

        results.extend(list(zip(y_true, y_pred)))

    y_true, y_pred = zip(*results)
    auc = roc_auc_score(y_true, y_pred)

    return losses.avg, auc

def evaluate(model, device, data_loader, criterion):

    losses = AverageMeter()
    results = []
    with torch.no_grad():
        model.eval()
        for i, (input, target) in enumerate(data_loader):

            if isinstance(input, tuple):
                input = tuple([e.to(device) if type(e) == torch.Tensor else e for e in input])
            else:
                input = input.to(device)

            target = target.to(device)

            output = model(input)
            loss = criterion(output, target)
            losses.update(loss.item(), target.size(0))

            y_true = target.detach().to('cpu').numpy().tolist()
            y_pred = nn.Softmax(1)(output).detach().to('cpu').numpy()[:,1].tolist()

            results.extend(list(zip(y_true, y_pred)))
        y_true, y_pred = zip(*results)
        auc = roc_auc_score(y_true, y_pred)

    return losses.avg, auc

