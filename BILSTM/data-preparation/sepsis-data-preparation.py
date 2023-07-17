import os
import pickle
import pandas as pd
from datetime import datetime
from datetime import timedelta
import numpy as np
import random

fix_seed = 42
random.seed(fix_seed)
np.random.seed(fix_seed)

PATH_TRAIN = "/home/ubuntu/mhb/data_note/train/train.csv"
PATH_VALIDATION = "/home/ubuntu/mhb/data_note/val/val.csv"
PATH_TEST = "/home/ubuntu/mhb/data_note/test/test.csv"
PATH_OUTPUT = "/home/ubuntu/myb/main-sepsis/data/processed/"


# 取seq对应的label
def create_dataset(path, observation_window=6):
    """
    :param path: path to the directory contains raw files.
    :param observation window: time interval we will use to identify relavant events
    :param prediction window: a fixed time interval that is to be used to make the prediction
    :return: List(pivot vital records), List(labels), time sequence data as a List of List of List.
    """
    seqs = []
    labels = []
    prediction_window = t
    # load data from csv;
    df = pd.read_csv(path)

    # construct features
    grouped_df = df.groupby('icustay_id')
    for name, group in grouped_df:
        # calculate the index_hour
        # for the patients who have the sepsis, index hour is #prediction_window hours prior to the onset time
        # for the patients who don't have the sepsis, index hour is the last event time
        if group.iloc[-1,-1] == 1:
            index_hour = datetime.strptime(group.iloc[-1,1], '%Y-%m-%d %H:%M:%S') - timedelta(hours=prediction_window)
        else:
            index_hour = datetime.strptime(group.iloc[-1,1], '%Y-%m-%d %H:%M:%S')
        index_hour = datetime.strptime(group.iloc[-1, 1], '%Y-%m-%d %H:%M:%S') - timedelta(hours=prediction_window)

        # get the date in observation window
        group['chart_time'] = pd.to_datetime(group['chart_time'])
        filterd_group = group[(group['chart_time'] >= (index_hour - timedelta(hours=observation_window))) & (group['chart_time'] <= index_hour)]
        # construct the records seqs and label seqs
        data = filterd_group.iloc[:, 3:-1]  # 取特征
        record_seqs = []
        for i in range(0, data.shape[0], 1):
            record_seqs.append(data.iloc[i].tolist())

        if len(record_seqs) != 0:
            seqs.append(record_seqs)
            labels.append(group.iloc[-1, -1])

    return seqs, labels  # 每一个seq对应一个lable


def main():
    os.makedirs(PATH_OUTPUT, exist_ok=True)
    # Train set
    print("Construct train set")
    train_seqs, train_labels = create_dataset(PATH_TRAIN)

    # 随机上采样
    # # 计算标签为1和标签为0的数量
    # count_1 = sum(train_labels)
    # count_0 = len(train_labels) - count_1
    #
    # # 计算需要上采样的数量
    # n = count_0 - count_1
    #
    # # 拆分训练集标签为1和标签为0的序列和标签
    # seq_1 = [train_seqs[i] for i in range(len(train_labels)) if train_labels[i] == 1]
    # labels_1 = [train_labels[i] for i in range(len(train_labels)) if train_labels[i] == 1]
    # seq_0 = [train_seqs[i] for i in range(len(train_labels)) if train_labels[i] == 0]
    # labels_0 = [train_labels[i] for i in range(len(train_labels)) if train_labels[i] == 0]
    #
    # # 对标签为1的序列和标签进行随机上采样
    # if n > 0:
    #     new_seq_1 = list(np.random.choice(seq_1, size=n, replace=True))
    #     new_labels_1 = [1] * n
    #     # 将上采样后的序列和标签与标签为0的序列和标签合并
    #     balanced_train_seqs = seq_0 + new_seq_1 + seq_1
    #     balanced_train_labels = labels_0 + new_labels_1 + labels_1
    # else:
    #     # 如果不需要上采样，则直接将标签为0和标签为1的序列和标签合并
    #     balanced_train_seqs = seq_0 + seq_1
    #     balanced_train_labels = labels_0 + labels_1
    #
    # a = balanced_train_labels.count(0)
    # b = balanced_train_labels.count(1)
    pickle.dump(train_seqs, open(os.path.join(PATH_OUTPUT, "sepsis.seqs.train"), 'wb'), pickle.HIGHEST_PROTOCOL)
    pickle.dump(train_labels, open(os.path.join(PATH_OUTPUT, "sepsis.labels.train"), 'wb'), pickle.HIGHEST_PROTOCOL)

    # Validation set
    print("Construct validation set")
    train_seqs, train_labels = create_dataset(PATH_VALIDATION)

    pickle.dump(train_seqs, open(os.path.join(PATH_OUTPUT, "sepsis.seqs.validation"), 'wb'), pickle.HIGHEST_PROTOCOL)
    pickle.dump(train_labels, open(os.path.join(PATH_OUTPUT, "sepsis.labels.validation"), 'wb'), pickle.HIGHEST_PROTOCOL)

    # Test set
    print("Construct test set")
    train_seqs, train_labels = create_dataset(PATH_TEST)

    pickle.dump(train_seqs, open(os.path.join(PATH_OUTPUT, "sepsis.seqs.test"), 'wb'), pickle.HIGHEST_PROTOCOL)
    pickle.dump(train_labels, open(os.path.join(PATH_OUTPUT, "sepsis.labels.test"), 'wb'), pickle.HIGHEST_PROTOCOL)

    print("Complete!")

if __name__ == '__main__':
    main()
