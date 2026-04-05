import torch
import torch.nn as nn


class CRNN(nn.Module):

    def __init__(self, num_classes=2):
        super(CRNN, self).__init__()

        self.cnn = nn.Sequential(

            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d((2,1))
        )

        self.freq_pool = nn.AdaptiveAvgPool2d((1, None))

        self.lstm = nn.LSTM(
            input_size=256,
            hidden_size=128,
            num_layers=2,
            batch_first=True,
            bidirectional=True
        )

        self.fc = nn.Sequential(
            nn.Linear(256,128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128,num_classes)
        )


    def forward(self, x):

        x = self.cnn(x)

        x = self.freq_pool(x)

        x = x.squeeze(2)

        x = x.permute(0,2,1)

        x,_ = self.lstm(x)

        x = x[:,-1,:]

        x = self.fc(x)

        return x