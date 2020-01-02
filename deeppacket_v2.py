__author__ = 'LiuYun'
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, TensorDataset, random_split
import numpy as np
import pickle
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from torch.utils.tensorboard import SummaryWriter
import time


os.environ["CUDA_VISIBLE_DEVICES"] = "1, 2, 3, 4"
# model
class Onedconv(nn.Module):
    def __init__(self):
        super(Onedconv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 100, 10, 4),
            nn.ReLU(),
            nn.Conv2d(100, 200, 10, 4),
            nn.ReLU(),
        )
        self.fc = nn.Sequential(
            nn.Linear(72800, 3000),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(3000, 600),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(600, 100),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(100, 20),
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


class Clstm(nn.Module):
    def __init__(self):
        super(Clstm, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, 64, 20, 10),
            nn.ReLU(),
            nn.Conv1d(64, 128, 20, 10),
            nn.ReLU(),
        )
        self.lstm = nn.LSTM(1664, 1664, 2, dropout=0.5)
        self.fc = nn.Linear(1664, 10)


    def forward(self, x):
        x = x.view(-1, 1, 1500)
        conv = self.conv(x)
        # print(conv.shape)
        emb = conv.view(-1, 100, 1664)
        self.lstm.flatten_parameters()
        lstm, _ = self.lstm(emb)
        lstm_last = lstm[:, -1]
        fc = self.fc(lstm_last)
        return fc



def train(train_loader, dev_loader, label_name, net, optimizer, criterion, epochs, batch_size, LR, is_train, scheduler):
    # timedate = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    datetime = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    if is_train:
        train_writer = SummaryWriter("../runs/train/" + datetime)
        dev_writer = SummaryWriter("../runs/dev/" + datetime)
        train_gap = 100
        dev_gap = 200
        if torch.cuda.is_available():
            net = net.cuda()
            if torch.cuda.device_count() > 1:
                net = nn.DataParallel(net)
        train_loss = 0
        for epoch in range(epochs):
            step_num = len(train_loader)
            # for step in range(step_num):
            for step, data in enumerate(train_loader):
                net.train()
                global_step = step + step_num * epoch
                batch_x, batch_y, flow_idx = data
                batch_x = batch_x.view(-1, 1, 100, 1500).float()
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
                    cm = confusion_matrix(batch_y, pre, labels=list(range(len(label_name))))
                    print(cm)
                    train_writer.add_scalar("loss", train_loss / train_gap, global_step)
                    train_writer.add_scalar("acc", acc, global_step)
                    train_writer.add_pr_curve("pr", batch_y, pre, global_step)
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
                    pre_all = torch.zeros(len(dev_loader.dataset))
                    label_all = torch.zeros(len(dev_loader.dataset))
                    run_loss = 0
                    start = 0
                    for i, data in enumerate(dev_loader):
                        dev_x, dev_y, flow_idx = data
                        dev_x = dev_x.view(-1, 1, 100, 1500).float()
                        dev_y = dev_y.long()
                        if torch.cuda.is_available():
                            dev_x = dev_x.cuda()
                            dev_y = dev_y.cuda()
                        with torch.no_grad():
                            dev_out = net(dev_x)
                            dev_loss = criterion(dev_out, dev_y)
                            run_loss += dev_loss.item()
                        dev_pre = torch.max(dev_out, 1)[1]
                        pre_all[start: start + len(dev_y)] = dev_pre.cpu()
                        label_all[start: start + len(dev_y)] = dev_y.cpu()
                        start = start + len(dev_y)
                    print("test loss at step %d is : %f" % (global_step, run_loss / len(dev_loader)))
                    dev_acc = accuracy_score(label_all, pre_all)
                    scheduler.step(dev_acc)
                    dev_cm = confusion_matrix(label_all, pre_all, labels=list(range(len(label_name))))
                    print(dev_cm)
                    dev_writer.add_scalar("loss", dev_loss, global_step)
                    dev_writer.add_scalar("acc", dev_acc, global_step)
                    dev_writer.add_pr_curve("pr", label_all, pre_all, global_step)
                    cm_df = pd.DataFrame(dev_cm, index=label_name, columns=label_name)
                    fig_cm = plt.figure()
                    sns.heatmap(cm_df, annot=True, fmt="d")
                    plt.title('Accuracy:{0:.3f}'.format(dev_acc))
                    dev_writer.add_figure('cm', fig_cm, global_step)
                if global_step % 1000 == 999:
                    torch.save(net.state_dict(), "../save/%d" % global_step)

        train_writer.close()
        dev_writer.close()
    else:
        test_writer = SummaryWriter("../runs/test/" + datetime)
        for _, _, file in os.walk('../save'):
            file_num = len(file)
        save_step = 999
        if torch.cuda.is_available():
            net = net.cuda()
            if torch.cuda.device_count() > 1:
                net = nn.DataParallel(net)
        for i in range(file_num - 1):
            net.load_state_dict(torch.load('../save/%d' % save_step))
            net.eval()
            pre_all = torch.zeros(len(dev_loader.dataset))
            label_all = torch.zeros(len(dev_loader.dataset))
            run_loss = 0
            start = 0
            error = {}
            for i, data in enumerate(dev_loader):
                dev_x, dev_y, flow_idx = data
                dev_x = dev_x.view(-1, 1, 100, 1500).float()
                dev_y = dev_y.long()
                if torch.cuda.is_available():
                    dev_x = dev_x.cuda()
                    dev_y = dev_y.cuda()
                with torch.no_grad():
                    dev_out = net(dev_x)
                    dev_loss = criterion(dev_out, dev_y)
                    run_loss += dev_loss.item()
                dev_pre = torch.max(dev_out, 1)[1]
                pre_all[start: start + len(dev_y)] = dev_pre.cpu()
                label_all[start: start + len(dev_y)] = dev_y.cpu()
                start = start + len(dev_y)
                for idx, y_label in enumerate(dev_y):
                    if y_label != dev_pre[idx]:
                        if str(flow_idx[idx].item()) in error.keys():
                            error[str(flow_idx[idx].item())] += 1
                        else:
                            error[str(flow_idx[idx].item())] = 1
            dev_acc = accuracy_score(label_all, pre_all)
            print("test loss at step %d is : %f, acc: %f" % (save_step, run_loss / len(dev_loader), dev_acc))
            print(sorted(error.items(), key=lambda d: d[0]))
            dev_cm = confusion_matrix(label_all, pre_all, labels=list(range(len(label_name))))
            print(dev_cm)
            test_writer.add_scalar("loss", dev_loss, save_step)
            test_writer.add_scalar("acc", dev_acc, save_step)
            test_writer.add_pr_curve("pr", label_all, pre_all, save_step)
            cm_df = pd.DataFrame(dev_cm, index=label_name, columns=label_name)
            fig_cm = plt.figure()
            sns.heatmap(cm_df, annot=True, fmt="d")
            plt.title('Accuracy:{0:.3f}'.format(dev_acc))
            test_writer.add_figure('cm', fig_cm, save_step)
            save_step += 1000
            # error_pd = pd.DataFrame(data=error)
            # error_pd.to_csv("./error/error.csv")
        test_writer.close()


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


def generate_data_test(data_dir, marks, l2n, classes, class_num=10, chunk_size=100):
    data_x = []
    data_y = []
    num_arr = [0 for i in range(classes)]
    for _, _, csv_files in os.walk(data_dir):
        for i, csv in enumerate(csv_files):
            print(csv)
            time1 = time.time()
            mark = pd.read_csv(data_dir + csv, nrows=1, header=None).values.tolist()[0][-1]
            time2 = time.time()
            print("mark time:" + str(time2 - time1))
            if (mark in marks) and (num_arr[l2n[str(mark)]] < class_num):
                chunks = pd.read_csv(data_dir + csv, header=None, chunksize=chunk_size)
                time3 = time.time()
                print("chunk time:" + str(time3 - time2))
                for idx, chunk in enumerate(chunks):
                    if num_arr[l2n[str(mark)]] < class_num:
                        time4 = time.time()
                        data = []
                        chunk = chunk.values.tolist()
                        for idx, line in enumerate(chunk):
                            data.append(line[:-1])
                            if (idx + 1) % 100 == 0:
                                data_x.append(data)
                                data_y.append(l2n[str(line[-1])])
                                data = []
                                num_arr[l2n[str(line[-1])]] += 1
                        time5 = time.time()
                        print("100 time:" + str(time5 - time4))
                    else:
                        break

        break
    print(num_arr)
    tensor_x = np.array(data_x)
    tensor_y = np.array(data_y)
    return tensor_x, tensor_y

class myDataset(Dataset):

    def __init__(self, file, l2n):
        self.file = file
        self.l2n = l2n
        self.data = pd.read_csv(self.file, header=None).values
        self.total_len = int(self.data.shape[0]/100)
        print(self.total_len)

    def __len__(self):
        return self.total_len

    def __getitem__(self, item):
        data = self.data[item*100: item*100 + 100, :1500]
        mask = self.data[item*100][-1]
        label = self.l2n[str(mask)]
        return data, label


class myDatasetNew(Dataset):

    def __init__(self, file, rate, is_train):
        self.file = file
        self.train_rate = rate
        self.data = None
        self.is_train = is_train
        for _, _, filelist in os.walk(self.file):
            for filename in filelist:
                if filename.split(".")[-1] == "csv":
                    data = pd.read_csv(self.file + filename, header=None).values
                    length = int(len(data) * self.train_rate)
                    if self.data is None:
                        if self.is_train:
                            self.data = data[:length]
                        else:
                            self.data = data[length:]
                    else:
                        if self.is_train:
                            self.data = np.concatenate((self.data, data[:length]), axis=0)
                        else:
                            self.data = np.concatenate((self.data, data[length:]), axis=0)
            break
        self.total_len = int(self.data.shape[0]/100)
        print(self.total_len)

    def __len__(self):
        return self.total_len

    def __getitem__(self, item):
        data = self.data[item*100: item*100 + 100, 2:]
        label = self.data[item*100][0]
        # label = 3
        idx = self.data[item*100][1]
        return data, label, idx





def main():
    is_train = 0
    data_dir = "../data/flow/12.4/"
    test_dir = "../data/test/"
    marks = [200, 244, 245, 246, 247, 248, 249, 251, 252]
    l2n = {"200": 1, "244": 6, "245": 3, "246": 3, "247": 5, "248": 4, "249": 2, "251": 0, "252": 7}
    train_label = ['others', 'Baidu', 'QQ', 'Wechat', 'Weiyun', 'Mail']#, 'ssh'
    # test_label = ['baidu', 'weiyu', 'qq', 'tim', 'wechat', 'https']
    test_label = []
    label_name = train_label + test_label
    if is_train:
        fullset = myDatasetNew(data_dir, 0.2, is_train)
        train_size = int(0.9 * len(fullset))
        test_size = len(fullset) - train_size
        trainset, devset = random_split(fullset, [train_size, test_size])

    else:
        testset = myDatasetNew(data_dir, 0.2, is_train)
        print(len(testset))
    LR = 1e-3
    batch_size = 128
    epochs = 100
    # net = Onedconv()
    net = Clstm()
    optimizer = optim.Adam(net.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3, verbose=True)
    criterion = nn.CrossEntropyLoss()
    if is_train:
        train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=8)
        dev_loader = DataLoader(devset, batch_size=batch_size, shuffle=True, num_workers=8)
        train(train_loader, dev_loader, label_name, net, optimizer, criterion, epochs, batch_size, LR, is_train, scheduler)
    else:
        test_loader = DataLoader(testset, batch_size=batch_size, shuffle=True, num_workers=8)
        train(test_loader, test_loader, label_name, net, optimizer, criterion, epochs, batch_size, LR, is_train, scheduler)


if __name__ == '__main__':
    main()
