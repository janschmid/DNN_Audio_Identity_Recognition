import torch
from torch import nn


class WhichPodcasterLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, num_labels = 2, max_sample_length=100):
        super().__init__()
        self.hs = hidden_size
        self.nl = num_layers
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=1, batch_first=True)
        self.softmax = nn.LogSoftmax(dim=1)
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
            # nn.Dropout(0.3),
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

class WhichPodcasterCNN(nn.Module):
    def __init__(self, num_labels = 2):
        super().__init__()
        self.softmax = nn.LogSoftmax(dim=0)

        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 3),
            nn.ReLU(),
            nn.MaxPool2d(3),
            nn.Conv2d(32, 64, 10),
            nn.ReLU(),
            nn.MaxPool2d(3),
            nn.LazyBatchNorm2d(),
            nn.Dropout2d(0.3),
            nn.ReLU(),
            # nn.MaxPool2d(3)
        )
        self.flatten=  nn.Sequential(
            nn.Flatten(),
            nn.ReLU())
        self.linear = nn.Sequential(
            nn.LazyBatchNorm1d(),
            nn.Dropout(0.3),
            nn.LazyLinear(1000),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.LazyLinear(10),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.LazyLinear(num_labels)
        )

    def forward(self, x):
        # x = x.unsqueeze(1)
        x = self.conv(x)
        x = self.flatten(x)
        x = self.linear(x)
        x = self.softmax(x)
        x = x.unsqueeze(2)
        return x


    
