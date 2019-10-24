__author__ = 'LiuYun'
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, TensorDataset
import numpy as np
import pickle
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from torch.utils.tensorboard import SummaryWriter
import time
import os

# model
class Onedconv(nn.Module):
    def __init__(self):
        super(Onedconv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, 200, 5, 2),
            nn.ReLU(),
            nn.Conv1d(200, 100, 4, 1),
            nn.ReLU(),
            nn.MaxPool1d(2, 2)
        )
        self.fc = nn.Sequential(
            nn.Linear(37200, 600),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(600, 300),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(300, 100),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(100, 16),
            # nn.ReLU(),
            # nn.Dropout(0.25),
            # nn.Linear(300, 200),
            # nn.ReLU(),
            # nn.Dropout(0.25),
            # nn.Linear(200, 100),
            # nn.ReLU(),
            # nn.Dropout(0.25),
            # nn.Linear(100, 50),
            # nn.ReLU(),
            # nn.Dropout(0.25),
            # nn.Linear(50, 17),
        )

    def forward(self, x):
        conv = self.conv(x)
        fc = self.fc(conv.view(conv.size(0), -1))
        return fc


def train(train_loader, dev_loader, label_name, net, epochs, batch_size, LR, is_train):
    # timedate = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    datetime = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    if is_train:
        train_writer = SummaryWriter("./runs/train/" + datetime)
        dev_writer = SummaryWriter("./runs/dev/" + datetime)
        train_gap = 50
        dev_gap = 500
        if torch.cuda.is_available():
            net = net.cuda()
            if torch.cuda.device_count() > 1:
                net = nn.DataParallel(net)
        optimizer = optim.Adam(net.parameters(), lr=LR)
        criterion = nn.CrossEntropyLoss()
        train_loss = 0
        for epoch in range(epochs):
            step_num = len(train_loader)
            # for step in range(step_num):
            for step, data in enumerate(train_loader):
                net.train()
                global_step = step + step_num * epoch
                batch_x, batch_y = data
                batch_x = batch_x.view(-1, 1, 1500).float()
                batch_y = batch_y.long()
                if torch.cuda.is_available():
                    batch_x = batch_x.cuda()
                    batch_y = batch_y.cuda()
                optimizer.zero_grad()
                output = net(batch_x)
                loss = criterion(output, batch_y)
                train_loss += loss.item()
                loss.backward()
                optimizer.step()
                if global_step % train_gap == train_gap - 1:
                    print("train loss at step %d is : %f" % (global_step, train_loss / train_gap))
                    pre = torch.max(output, 1)[1]
                    if torch.cuda.is_available():
                        batch_y = batch_y.cpu()
                        pre = pre.cpu()
                    acc = accuracy_score(batch_y, pre)
                    cm = confusion_matrix(batch_y, pre)
                    print(cm)
                    train_writer.add_scalar("loss", train_loss / train_gap, global_step)
                    train_writer.add_scalar("acc", acc, global_step)
                    train_writer.add_pr_curve("pr", batch_y, pre, global_step)
                    if len(cm) == len(label_name):
                        cm_df = pd.DataFrame(cm, index=label_name, columns=label_name)
                        fig_cm = plt.figure()
                        sns.heatmap(cm_df, annot=True, fmt="d")
                        plt.title('Accuracy:{0:.3f}'.format(acc))
                        # plt.ylabel('True label')
                        # plt.xlabel('Predicted label')
                        train_writer.add_figure('cm', fig_cm, global_step)
                    train_loss = 0
                if global_step % dev_gap == dev_gap - 1:
                    net.eval()
                    for i, data in enumerate(dev_loader):
                        dev_x, dev_y = data
                        dev_x = dev_x.view(-1, 1, 1500).float()
                        dev_y = dev_y.long()
                        if torch.cuda.is_available():
                            dev_x = dev_x.cuda()
                            dev_y = dev_y.cuda()
                        with torch.no_grad():
                            dev_out = net(dev_x)
                            dev_loss = criterion(dev_out, dev_y)
                        print("dev loss at step %d is : %f" % (global_step, dev_loss))
                        dev_pre = torch.max(dev_out, 1)[1]
                        if torch.cuda.is_available():
                            dev_y = dev_y.cpu()
                            dev_pre = dev_pre.cpu()
                        dev_acc = accuracy_score(dev_y, dev_pre)
                        dev_cm = confusion_matrix(dev_y, dev_pre)
                        print(dev_cm)
                        dev_writer.add_scalar("loss", dev_loss, global_step)
                        dev_writer.add_scalar("acc", dev_acc, global_step)
                        dev_writer.add_pr_curve("pr", dev_y, dev_pre, global_step)
                        if len(dev_cm) == len(label_name):
                            cm_df = pd.DataFrame(dev_cm, index=label_name, columns=label_name)
                            fig_cm = plt.figure()
                            sns.heatmap(cm_df, annot=True, fmt="d")
                            plt.title('Accuracy:{0:.3f}'.format(dev_acc))
                            # plt.ylabel('True label')
                            # plt.xlabel('Predicted label')
                            dev_writer.add_figure('cm', fig_cm, global_step)
        train_writer.close()
        dev_writer.close()


def generate_data(data_dir, skip, nrows):
    data_x = []
    data_y = []
    label_name = []
    for _, _, csv_files in os.walk(data_dir):
        for i, csv in enumerate(csv_files):
            print(csv)
            df = pd.read_csv(data_dir + csv, header=None, skiprows=skip, nrows=nrows).values.tolist()
            data_x += df
            print(len(data_x))
            label = [i for x in range(0, nrows)]
            data_y += label
            csv = csv.split(".")
            label_name.append(csv[0])
    tensor_x = torch.from_numpy(np.array(data_x))
    tensor_y = torch.from_numpy(np.array(data_y))
    dataset = TensorDataset(tensor_x, tensor_y)
    return dataset, label_name


def main():
    data_dir = "data/noip/"
    trainset, label_name = generate_data(data_dir, 0, 20000)
    print(label_name)
    devset, _ = generate_data(data_dir, 20000, 1000)
    LR = 1e-3
    batch_size = 256
    epochs = 200
    net = Onedconv()
    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    dev_loader = DataLoader(devset, batch_size=1000*len(label_name), shuffle=True)
    train(train_loader, dev_loader, label_name, net, epochs, batch_size, LR, 1)


main()
