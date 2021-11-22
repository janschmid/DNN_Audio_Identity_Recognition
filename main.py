# from _typeshed import Self
import os
from google_drive_downloader import GoogleDriveDownloader as gdd
from torch._C import set_flush_denormal
from torch.utils.data import dataset
from torchaudio import datasets
import torch
from torch import nn
import numpy as np
import torchaudio
from custom_dataset import AudioDataset
import tqdm
import copy

root_dir = file_path = os.path.dirname(os.path.realpath(__file__))
datasetDir = os.path.join(root_dir, "dataset_100msec_chunks")
os.makedirs(datasetDir, exist_ok=True)
gdd.download_file_from_google_drive('1Plh2FwwShj9fq-lCZaFqT04D7wi5F5RN', os.path.join("dataset", "podcast_audio.zip"), unzip=True)
    

class WhichPodcaster(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, num_labels = 2, max_sample_length=100):
        super().__init__()
        self.hs = hidden_size
        self.nl = num_layers
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=1, batch_first=True)
        self.softmax = nn.LogSoftmax()
        self.fc = nn.Sequential(
        nn.Linear(hidden_size,32),
        nn.ReLU(),
        nn.BatchNorm1d(max_sample_length),
        nn.Linear(32,num_labels)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(max_sample_length,64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64,16),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(16,1)
            )
        

    def forward(self, x):
        x = x.transpose(2,1)
        h0 = torch.zeros(1, x.shape[0], self.hs).cuda()
        c0 = torch.zeros(1, x.shape[0], self.hs).cuda()
        x, _ = self.lstm(x, (h0, c0))
        x = self.fc(x)
        x = x.transpose(2,1)
        x = self.fc2(x)
        x = self.softmax(x)

        return x

def train(model, f_train_data, f_test_data):
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.NLLLoss()

    train_losses,test_losses = [],[]
    train_acc,test_acc = [],[]
    for epoch in range(50):
        model.train()
        loss_epoch,acc_epoch = 0,0
        num_batches = 0
        total_train = 0
        total_test = 0
        for x, y in f_train_data:
            x,y = x.cuda(),y.cuda()
            yhat = model(x)
            yu = y.unsqueeze(1).float()
            # yhat = yhat.argmax(dim=1).float()
            loss = criterion(yhat, yu.long())

            opt.zero_grad()
            loss.backward()
            opt.step()

            loss_epoch += loss.item()
            acc_epoch += (torch.argmax(yhat, dim=1) == y.unsqueeze(1)).sum().item()
            num_batches += 1
            total_train += x.shape[0]

        train_losses.append(loss_epoch / num_batches)
        train_acc.append(acc_epoch / total_train)

        model.eval()
        loss_epoch,acc_epoch = 0,0
        num_batches = 0
        for x, y in f_test_data:
            x,y = x.cuda(),y.cuda()
            yhat = model(x)
            yu = y.unsqueeze(1).float()
            loss = criterion(yhat, yu.long())

            loss_epoch += loss.item()
            acc_epoch += (torch.argmax(yhat, dim=1) == y.unsqueeze(1)).sum().item()
            num_batches += 1
            total_test += x.shape[0]
        
        test_losses.append(loss_epoch / num_batches)
        test_acc.append(acc_epoch / total_test)
        if test_acc[-1] == min(test_acc):
                      best_model = copy.deepcopy(model)
                      torch.save(best_model, 'WhichPodcaster.pth')
        print(f'Epoch {epoch}, train loss: {train_losses[-1]:.4f}, train acc: {train_acc[-1]:.4f}, test loss: {test_losses[-1]:.4f}, test acc: {test_acc[-1]:.4f}')


if __name__ == "__main__":
    max_sample_length = 10
    num_labels = 2

    data = AudioDataset(datasetDir, max_sample_length, transform=nn.Sequential(
    # torchaudio.transforms.MelSpectrogram(),
    # torchaudio.transforms.AmplitudeToDB(),
    torchaudio.transforms.MFCC(),
    # torchaudio.transforms.AmplitudeToDB(),
    ))

    num_train = len(data)
    indices = list(range(num_train))
    validation_size = 0.2
    split = int(np.floor(validation_size * num_train))
    np.random.shuffle(indices)
    from torch.utils.data.sampler import SubsetRandomSampler
    train_idx, test_idx = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)
    test_sampler = SubsetRandomSampler(test_idx)
    bs = 2000
    train_dataloader = torch.utils.data.DataLoader(data, sampler=train_sampler, batch_size=bs)
    test_dataloader = torch.utils.data.DataLoader(data, sampler=test_sampler, batch_size=bs)


    model = WhichPodcaster(len(data[0][0]), 400, num_labels=num_labels, max_sample_length=data.max_sample_length).cuda()
    train(model, train_dataloader, test_dataloader)
    


