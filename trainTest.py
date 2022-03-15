# from _typeshed import Self
import os
from google_drive_downloader import GoogleDriveDownloader as gdd
from torch._C import set_flush_denormal
from torch.utils.data import dataset
from torchaudio import datasets
import torch
from torch import nn
import numpy as np
from custom_dataset import AudioDataset, lambda_noise_transform
import copy
from torch.utils.tensorboard import SummaryWriter
from  time import strftime, gmtime


def train(model, f_train_data, f_test_data, model_name, num_epochs=100):
    writer = SummaryWriter(os.path.join(os.environ["miniproject_output_path"], 'runs',model_name))
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    if("n_lab: 3" in model_name):
        criterion = nn.CrossEntropyLoss(weight=torch.tensor([1.8, 1., 1.]).cuda(),reduction='sum')
    else:
        criterion = nn.CrossEntropyLoss(weight=torch.tensor([1., 1.]).cuda(),reduction='sum')
    train_losses,test_losses = [],[]
    train_acc,test_acc = [],[]
    for epoch in range(num_epochs):
        model.train()
        loss_epoch,acc_epoch = 0,0
        num_batches = 0
        total_train = 0
        total_test = 0
        for x, y in f_train_data:
            x,y = x.cuda(),y.cuda()
            yhat = call_model(model, x, model_name)
            yu = y.unsqueeze(1).long()

            # yhat = yhat.argmax(dim=1).float()
            loss = criterion(yhat, yu)

            opt.zero_grad()
            loss.backward()
            opt.step()

            loss_epoch += loss.item()
            acc_epoch += (torch.argmax(yhat, dim=1) == y.unsqueeze(1)).sum().item()
            num_batches += 1
            total_train += x.shape[0]
        writer.add_scalar("Train/loss", loss_epoch/num_batches, epoch+1)
        writer.add_scalar("Train/acc", acc_epoch/total_train, epoch+1)
        train_losses.append(loss_epoch / num_batches)
        train_acc.append(acc_epoch / total_train)

        model.eval()
        loss_epoch,acc_epoch = 0,0
        num_batches = 0
        for x, y in f_test_data:
            x,y = x.cuda(),y.cuda()
            yhat = call_model(model, x, model_name)
            yu = y.unsqueeze(1).float()
            loss = criterion(yhat, yu.long())

            loss_epoch += loss.item()
            acc_epoch += (torch.argmax(yhat, dim=1) == y.unsqueeze(1)).sum().item()
            num_batches += 1
            total_test += x.shape[0]
        
        test_losses.append(loss_epoch / num_batches)
        test_acc.append(acc_epoch / total_test)
        writer.add_scalar("Test/loss", loss_epoch/num_batches, epoch+1)
        writer.add_scalar("Test/acc", acc_epoch/total_test, epoch+1)
        if test_acc[-1] == max(test_acc):
            print("Saving best model of epoch: {0}".format(epoch))
            best_model = copy.deepcopy(model)
            torch.save(best_model, os.path.join(os.environ["miniproject_output_path"] ,model_name+'.pth'))
        print(f'Epoch {epoch}, train loss: {train_losses[-1]:.4f}, train acc: {train_acc[-1]:.4f}, test loss: {test_losses[-1]:.4f}, test acc: {test_acc[-1]:.4f}')

def test(model, test_data, results, model_name):
    model.eval()
    for x,y in test_data:
        x = x.cuda()
        y_hat = call_model(model, x, model_name)
        results.append(torch.argmax(y_hat, dim=1).item())

def call_model(model, x, model_name):
    if not 'LSTM' in model_name:
        x = x.unsqueeze(1).cuda()
    if 'RESNET' in model_name:
        x = x.repeat(1,3,1,1).cuda()
        yhat = model(x).unsqueeze(2)
    else:
        yhat = model(x)
    return yhat
            