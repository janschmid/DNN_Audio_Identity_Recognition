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
import torchvision
from  trainTest import train,test
from custom_dataset import AudioDataset
import tqdm
import copy
from torch.utils.tensorboard import SummaryWriter
from stats_utils import plot_confusionmat
from Model_Podcasters import WhichPodcasterLSTM, WhichPodcasterCNN
from torch.utils.data.sampler import SubsetRandomSampler
from  time import strftime, gmtime

root_dir = file_path = os.path.dirname(os.path.realpath(__file__))
os.makedirs("dataset", exist_ok=True)
gdd.download_file_from_google_drive('1Plh2FwwShj9fq-lCZaFqT04D7wi5F5RN', os.path.join("dataset", "podcast_audio.zip"), unzip=True)
from enum import Enum

class ModelType(Enum):
    LSTM=0,
    CNN=1,
    RESNET=2,

def load_dataset(modelType, max_sample_length):
    datasetDir = "/home/jschm20/DNN/Miniproject/dataset_1000msec_chunks"

    if modelType == ModelType.LSTM:
        transform = nn.Sequential(
            torchaudio.transforms.MFCC(melkwargs={"f_min": 100}),  # melkwargs={"n_fft": 600}
        )
    if modelType == ModelType.CNN:
         transform = nn.Sequential(
            torchaudio.transforms.MFCC(melkwargs={"f_min": 100}),  # melkwargs={"n_fft": 600}
            # torchaudio.transforms.AmplitudeToDB(),
            torchvision.transforms.Resize((128, 128)),
        )
    if modelType == ModelType.RESNET:
         transform = nn.Sequential(
            torchaudio.transforms.MFCC(melkwargs={"f_min": 100}),  # melkwargs={"n_fft": 600}
            # torchaudio.transforms.AmplitudeToDB(),
            torchvision.transforms.Resize((224, 224)),
        )
        # # torchaudio.transforms.MelSpectrogram(),
        # # torchaudio.transforms.AmplitudeToDB(),
        # ))

    data = AudioDataset(datasetDir, max_sample_length, transform=transform)
    return data

def load_model(modelType, num_labels, data):
    if(modelType==ModelType.LSTM):
        return WhichPodcasterLSTM(len(data[0][0]), 400, num_labels=num_labels, max_sample_length=data.max_sample_length).cuda()
    if(modelType==ModelType.CNN):
        return WhichPodcasterCNN(num_labels=num_labels).cuda()
    if(modelType==ModelType.RESNET):
        resnet = torchvision.models.resnet50(pretrained=True)
        resnet.fc = nn.Sequential(nn.LazyLinear(100),
                                      nn.ReLU(),
                                      nn.Dropout(.2),
                                      nn.Linear(100, 3),
                                      nn.LogSoftmax(dim=1)
                                      )
        for param in resnet.parameters():
            param.requires_grad = False
        for param in resnet.fc.parameters():
            param.requires_grad = True
        return resnet.cuda()


if __name__ == "__main__":
    max_sample_length = 102
    num_labels = 3
    validation_size = 0.2
    model_type = ModelType.RESNET
    num_epochs = 50
    bs = 400
    data =load_dataset(model_type, max_sample_length)


    os.environ["miniproject_output_path"] = "/home/jschm20/DNN/Miniproject/output"
    model_name = 'WhichPodcaster_{0}_test'.format( model_type.name+" "+strftime("%a, %d %b %Y %H:%M:%S", gmtime()))
    num_train = len(data)
    indices = list(range(num_train))
    split = int(np.floor(validation_size * num_train))
    np.random.shuffle(indices)
    
    train_idx, test_idx = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)
    test_sampler = SubsetRandomSampler(test_idx)
    train_dataloader = torch.utils.data.DataLoader(data, sampler=train_sampler, batch_size=bs)
    test_dataloader = torch.utils.data.DataLoader(data, sampler=test_sampler, batch_size=bs)

    model = load_model(model_type, num_labels, data)
    
    train(model, train_dataloader, test_dataloader, model_name, num_epochs)
    
    bestModel = torch.load(os.path.join(os.environ["miniproject_output_path"],model_name)+'.pth')
    plot_confusionmat(bestModel, test_dataloader, model_name)
    
    # #### TEST ####
    # datasetDir = "Miniproject/youtubeDl/split_1s_chunks"
    # data = AudioDataset(datasetDir, max_sample_length, transform=nn.Sequential(
    # # torchaudio.transforms.MelSpectrogram(),
    # # torchaudio.transforms.AmplitudeToDB(),
    # torchaudio.transforms.MFCC(),
    # # torchaudio.transforms.AmplitudeToDB(),
    # ))

    # loader = torch.utils.data.DataLoader(data,  batch_size=1)
    # loaded_model = torch.load('WhichPodcaster2.pth').cuda()
    # torch.onnx.export(loaded_model, torch.zeros(1,40,100).cuda(), "WhichPodcaster2.onnx")
    # results = []
    # test(loaded_model, loader, results)
    # np.savetxt("1000ms_chunks_video.csv", results)
    


