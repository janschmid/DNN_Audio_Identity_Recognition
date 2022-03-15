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
from custom_dataset import AudioDataset, lambda_noise_transform
import tqdm
import copy
from torch.utils.tensorboard import SummaryWriter
from stats_utils import plot_confusionmat
from Model_Podcasters import WhichPodcasterLSTM, WhichPodcasterCNN
from torch.utils.data.sampler import SubsetRandomSampler
from  time import strftime, gmtime
import seaborn as sns

root_dir = file_path = os.path.dirname(os.path.realpath(__file__))
os.makedirs("dataset", exist_ok=True)
gdd.download_file_from_google_drive('1Plh2FwwShj9fq-lCZaFqT04D7wi5F5RN', os.path.join("dataset", "podcast_audio.zip"), unzip=True)
from enum import Enum

class ModelType(Enum):
    LSTM=0,
    CNN=1,
    RESNET=2,

def load_dataset(datasetDir, modelType, max_sample_length):
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
        transform = torchvision.transforms.Compose([
            # torchaudio.transforms.MFCC(melkwargs={"f_min": 100}),  # melkwargs={"n_fft": 600}
            # torchvision.transforms.Lambda(lambda x: x/255),
            # torchvision.transforms.Lambda(lambda x: x.astype(np.float32) / 255),
            torchaudio.transforms.MelSpectrogram(),
            torchaudio.transforms.AmplitudeToDB(),
            torchvision.transforms.Resize((224, 224)),
        ])
        # # torchaudio.transforms.AmplitudeToDB(),
        # ))

    data = AudioDataset(datasetDir, max_sample_length, transform=transform)
    import matplotlib.pyplot as plt
    sns.heatmap(data[10][0].numpy())
    plt.savefig(os.path.join(os.environ["miniproject_output_path"], modelType.name))
    return data, datasetDir.split('/')[-1]

def load_model(modelType, num_labels, data):
    if(modelType==ModelType.LSTM):
        return WhichPodcasterLSTM(len(data[0][0]), 400, num_labels=num_labels, max_sample_length=data.max_sample_length).cuda()
    if(modelType==ModelType.CNN):
        return WhichPodcasterCNN(num_labels=num_labels).cuda()
    if(modelType==ModelType.RESNET):
        # resnet = torchvision.models.resnet50(pretrained=True)
        resnet = torchvision.models.resnet18(pretrained=True)
        resnet.fc = nn.Sequential(nn.LazyLinear(100),
                                      nn.ReLU(),
                                      nn.Dropout(.3),
                                      nn.LazyBatchNorm1d(),
                                      nn.Linear(100, 3),
                                    #   nn.ReLU(),
                                      nn.Dropout(.1),
                                    #   nn.LazyBatchNorm1d(),
                                    #   nn.LogSoftmax(dim=1)
                                      )
        # for param in resnet.parameters():
        #     param.requires_grad = False
        # for param in resnet.fc.parameters():
        #     param.requires_grad = True
        return resnet.cuda()

def get_model_name(datasetName, model_type, num_labels):
    return 'WhichPodcaster_{0}_test'.format(datasetName+" "+ model_type.name+" n_lab: "+str(num_labels)+"  "+strftime("%a, %d %b %Y %H:%M:%S", gmtime()))


def execute_train(max_sample_length, num_labels, validation_size, model_type, num_epochs, bs):
    dataset_dir = "/home/jschm20/DNN/Miniproject/edited_dataset_1000msec_chunks"
    data, datasetName =load_dataset(dataset_dir, model_type, max_sample_length)


    model_name = get_model_name(datasetName, model_type, num_labels)
    num_train = len(data)
    indices = list(range(num_train))
    split = int(np.floor(validation_size * num_train))
    np.random.shuffle(indices)
    
    train_idx, test_idx = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)
    test_sampler = SubsetRandomSampler(test_idx)
    train_dataloader = torch.utils.data.DataLoader(copy.deepcopy(data), sampler=train_sampler, batch_size=bs)
    test_dataloader = torch.utils.data.DataLoader(data, sampler=test_sampler, batch_size=bs)

    train_dataloader.dataset.transform=torchvision.transforms.Compose([
        train_dataloader.dataset.transform,
        torchvision.transforms.Lambda(lambda x: lambda_noise_transform(x)),
    ])


    model = load_model(model_type, num_labels, data)
    train(model, train_dataloader, test_dataloader, model_name, num_epochs)
    
    bestModel = torch.load(os.path.join(os.environ["miniproject_output_path"],model_name)+'.pth')
    plot_confusionmat(bestModel, test_dataloader, model_name)


def execute_test(max_sample_length, num_labels, model_type):
     # #### TEST ####
    datasetDir = "Miniproject/youtubeDl/3minClip/split_1s_chunks"
    data, datasetName =load_dataset(datasetDir, model_type, max_sample_length)
    loader = torch.utils.data.DataLoader(data,  batch_size=1)
    loaded_model = torch.load('/home/jschm20/DNN/Miniproject/output/WhichPodcaster_edited_dataset_1000msec_chunks RESNET n_lab: 3  Mon, 13 Dec 2021 14:36:25_test.pth').cuda()
    model_name = get_model_name(datasetName, model_type, num_labels)
    exportPath = os.path.join(os.environ["miniproject_output_path"],model_name)
    results = []
    plot_confusionmat(loaded_model, loader, model_name)
    # test(loaded_model, loader, results, model_name)
    # np.savetxt(exportPath+".csv", results)
    # torch.onnx.export(loaded_model, torch.zeros(1,3,128,128).cuda(), exportPath+".onnx")


if __name__ == "__main__":
    os.environ["miniproject_output_path"] = "/home/jschm20/DNN/Miniproject/output"
    max_sample_length = 102
    num_labels = 3
    validation_size = 0.2
    model_type = ModelType.LSTM
    num_epochs = 40
    bs = 60
    execute_train(max_sample_length, num_labels, validation_size, model_type, num_epochs, bs)
    execute_test(max_sample_length, num_labels, model_type)
    
    
    
   
    


