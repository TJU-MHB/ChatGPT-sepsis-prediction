import os
import pickle
import torch
import random
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score,precision_recall_curve,auc,precision_score,recall_score,roc_curve,confusion_matrix,accuracy_score
from hyperopt import fmin, hp, tpe, STATUS_OK, Trials
from utils import train, evaluate, best_evaluate, bootstrap_confidence_interval, calculate_num_features, VisitSequenceWithLabelDataset, time_collate_fn
from model_bilstm import MyLSTM
from torch.utils.data.sampler import WeightedRandomSampler
from scipy import interp

fix_seed = 42
random.seed(fix_seed)
torch.manual_seed(fix_seed)
torch.cuda.manual_seed_all(fix_seed)
np.random.seed(fix_seed)
torch.multiprocessing.set_start_method('spawn')
os.environ["CUDA_VISIBLE_DEVICES"] = "0"



PATH_TRAIN_SEQS = "/home/ubuntu/mhb/main_sepsis/data/processed/sepsis.seqs.train"
PATH_TRAIN_LABELS = "/home/ubuntu/mhb/main_sepsis/data/sepsis.labels.train"
PATH_VALID_SEQS = "/home/ubuntu/mhb/main_sepsis/data/sepsis.seqs.validation"
PATH_VALID_LABELS = "/home/ubuntu/mhb/main_sepsis/data/sepsis.labels.validation"
PATH_TEST_SEQS = "/home/ubuntu/mhb/main_sepsis/data/sepsis.seqs.test"
PATH_TEST_LABELS = "/home/ubuntu/mhb/main_sepsis/data/sepsis.labels.test"



def lstm_fit(params):
    scalers = []
    models = []
    auc = []
    verbose = params['verbose']
    gru_input = params['gru_input']
    hidden_dim = params['hidden_dim']
    layer_size = params['layer_size']
    dropout = params['dropout']

    for fold, (train_index, valid_index) in enumerate(skf.split(train_val_seqs, train_val_labels)):
        x_train = np.array(train_val_seqs, dtype=object)[train_index].tolist()
        y_train = np.array(train_val_labels, dtype=object)[train_index].tolist()
        x_valid = np.array(train_val_seqs, dtype=object)[valid_index].tolist()
        y_valid = np.array(train_val_labels, dtype=object)[valid_index].tolist()

        train_dataset = VisitSequenceWithLabelDataset(x_train, y_train)
        valid_dataset = VisitSequenceWithLabelDataset(x_valid, y_valid)

        count_positive = 0
        count_negative = 0
        for data, label in train_dataset:
            if label == 1:
                count_positive += 1
            else:
                count_negative += 1

        weights = [len(train_dataset) / x for x in[count_positive if label == 1 else count_negative for data, label in train_dataset]]

        sampler = WeightedRandomSampler(weights, num_samples=len(train_dataset), replacement=True)
        train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, collate_fn=time_collate_fn,sampler=sampler,num_workers=NUM_WORKERS)
        valid_loader = DataLoader(dataset=valid_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=time_collate_fn, num_workers=NUM_WORKERS)

        model = MyLSTM(num_features, gru_input, hidden_dim, layer_size, dropout)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=LR)
        scheduler = ReduceLROnPlateau(optimizer, 'min', patience=3, factor=.2)
        model.to(device)
        criterion.to(device)

        best_val_auc = 0.0
        early_stopping = 0

        if verbose > 0: print(f'===> Training model for fold {fold}')
        for epoch in range(NUM_EPOCHS):
            train_loss, train_auc = train(model, device, train_loader, criterion, optimizer, epoch)
            valid_loss, valid_auc = evaluate(model, device, valid_loader, criterion)
            scheduler.step(-valid_auc)
            current_lr = optimizer.param_groups[0]['lr']

            if verbose > 0:
                print(f'Epoch: [{epoch}]\t'
                      'Train\t'
                      f'Loss {train_loss:.4f}\t'
                      f'AUC {train_auc:.4f}\t'
                      'Valid\t'
                      f'Loss {valid_loss:.4f}\t'
                      f'AUC {valid_auc:.4f}\t'
                      f'lr {current_lr}')

            is_best = valid_auc > best_val_auc
            if is_best:
                best_val_auc = valid_auc
                best_model = model
                early_stopping = 0
            else:
                early_stopping += 1
                if early_stopping >= 10:
                    print(f"early stopping at epoch {epoch}")
                    break

        models.append(best_model)
        auc.append(best_val_auc)

    return {
        'loss': -np.mean(auc),
        'scalers': scalers,
        'models': models,
        'params': params,
        'status': STATUS_OK,
    }


if __name__ == '__main__':

    print('===> Loading entire datasets')
    train_seqs = pickle.load(open(PATH_TRAIN_SEQS, 'rb'))
    train_labels = pickle.load(open(PATH_TRAIN_LABELS, 'rb'))
    valid_seqs = pickle.load(open(PATH_VALID_SEQS, 'rb'))
    valid_labels = pickle.load(open(PATH_VALID_LABELS, 'rb'))
    test_seqs = pickle.load(open(PATH_TEST_SEQS, 'rb'))
    test_labels = pickle.load(open(PATH_TEST_LABELS, 'rb'))

    train_val_seqs = train_seqs + valid_seqs
    train_val_labels = train_labels + valid_labels

    num_features = calculate_num_features(train_seqs)

    NUM_EPOCHS = 100
    BATCH_SIZE = 16
    USE_CUDA = True
    NUM_WORKERS = 0

    LR = 0.005

    device = torch.device("cuda" if torch.cuda.is_available() and USE_CUDA else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        print("use cuda")


    skf = StratifiedKFold(n_splits=5, random_state=42, shuffle=True)

    print('===> Hyperparameter tuning')
    lstm_params = {
        'gru_input': hp.choice('gru_input', [16, 32, 64, 128, 256]),
        'hidden_dim': hp.choice('hidden_dim', [8, 16, 32, 64, 128]),
        'layer_size': hp.choice('layer_size', [2, 4, 6, 8, 10]),
        'dropout': hp.choice('dropout', [.1, .2, .3, .4, .5]),
        'verbose': 1
    }
    trials = Trials()
    lstm_best = fmin(lstm_fit, lstm_params, algo=tpe.suggest, rstate=np.random.default_rng(42), max_evals=50,trials=trials)
    
    print('===> Best model:')
    print(trials.best_trial['result']['models'][0])
    print('===> Save model')
    for fold,model in enumerate(trials.best_trial['result']['models']):
        torch.save(model, os.path.join(f"/home/ubuntu/mhb/main_sepsis/out/output/BILSTM/lstm_model{fold}.pth"))

    print('===> Evaluate test data')
    y_prob_cv = np.empty((len(test_labels), 5))
    roc_auc = []
    pr_auc = []
    precisions = []
    recalls = []
    fprs = []
    confusion = []
    accs = []
    tnrs = []
    n_bootstrap = 1000
    confidence_level = 0.95
    roc_curves = []

    for fold in range(5):
        model = torch.load(os.path.join(f"/home/ubuntu/mhb/main_sepsis/out/output/BILSTM/lstm_model{fold}.pth"))

        test_dataset = VisitSequenceWithLabelDataset(test_seqs, test_labels)
        test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=time_collate_fn, num_workers=NUM_WORKERS)

        model.eval()
        results = []
        with torch.no_grad():
            for i, (input, target) in enumerate(test_loader):
                if isinstance(input, tuple):
                    input = tuple([e.to(device) if type(e) == torch.Tensor else e for e in input])
                else:
                    input = input.to(device)
                target = target.to(device)

                output = model(input)
                y_true = target.detach().to('cpu').numpy().tolist()
                y_pred = output.detach().to('cpu').max(1)[1].numpy().tolist()
                y_prob = nn.Softmax(1)(output).detach().to('cpu').numpy()[:, 1].tolist()

                results.extend(list(zip(y_true, y_prob,y_pred)))

            y_true, y_prob_cv[:, fold],y_pred = zip(*results)
            aucroc = roc_auc_score(y_true, y_prob_cv[:, fold])
            precision, recall, _ = precision_recall_curve(y_true, y_pred)
            pr = auc(recall, precision)
            pre = precision_score(y_true, y_pred)
            rec = recall_score(y_true, y_pred)
            cof = confusion_matrix(y_true,y_pred)
            acc = accuracy_score(y_true, y_pred)
            fpr, tpr, thresholds = roc_curve(y_true, y_prob_cv[:, fold])
            roc_curves.append((fpr, tpr))
            tnr = 1 - fpr[tpr >= 0.80][0]


            roc_auc.append(aucroc)
            pr_auc.append(pr)
            precisions.append(pre)
            recalls.append(rec)
            fprs.append(fpr[tpr >= 0.80][0])
            confusion.append(cof)
            accs.append(acc)
            tnrs.append(tnr)



    auroc_std = np.std(roc_auc)
    pr_auc_std = np.std(pr_auc)
    precision_std = np.std(precisions)
    recall_std = np.std(recalls)
    fpr_std = np.std(fprs)
    acc_std = np.std(accs)
    confusion_std = np.std(confusion, axis=0)

    print('test set Precision(PPV): {:.2f} ({:.2f}, {:.2f})'.format(np.mean(precisions), *bootstrap_confidence_interval(precisions, n_bootstrap, confidence_level)))
    print('test set Recall(sensitivity/TPR): {:.2f} ({:.2f}, {:.2f})'.format(np.mean(recalls), *bootstrap_confidence_interval(recalls, n_bootstrap, confidence_level)))
    print('test set FPR(1-TNR(specificity)): {:.2f} ({:.2f}, {:.2f})'.format(np.mean(fprs), *bootstrap_confidence_interval(fprs, n_bootstrap, confidence_level)))
    print('test set TNR(specificity): {:.2f} ({:.2f}, {:.2f})'.format(np.mean(tnrs), *bootstrap_confidence_interval(tnrs, n_bootstrap, confidence_level)))
    print('test set PR-AUC: {:.2f} ({:.2f}, {:.2f})'.format(np.mean(pr_auc), *bootstrap_confidence_interval(pr_auc, n_bootstrap, confidence_level)))
    print('test set ROC-AUC: {:.2f} ({:.2f}, {:.2f})'.format(np.mean(roc_auc), *bootstrap_confidence_interval(roc_auc, n_bootstrap, confidence_level)))
    print('test set ACC: {:.2f} ({:.2f}, {:.2f})'.format(np.mean(accs), *bootstrap_confidence_interval(accs, n_bootstrap, confidence_level)))
    print('test confusion matrix:  \n ', np.round(np.mean(confusion, axis=0)).astype(int))
    print('test confusion matrix std: \n', np.round(confusion_std))  
    print('confusion:\n',confusion)

  