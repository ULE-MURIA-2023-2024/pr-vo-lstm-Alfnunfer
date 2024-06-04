import torch
import torch.nn as nn
from torchvision.models import resnet50 as resnet
from torchvision.models import ResNet50_Weights as weights
from typing import Callable


class VisualOdometryModel(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        num_layers: int,
        bidirectional: bool = False,
        lstm_dropout: float = 0.2
    ) -> None:

        super(VisualOdometryModel, self).__init__()

        # Load pre-trained ResNet model
        self.cnn_model = resnet(weights=weights.DEFAULT)
        resnet_output = list(self.cnn_model.children())[-1].in_features
        self.cnn_model.fc = nn.Identity()

        # Freeze the weights of the ResNet layers
        for param in self.cnn_model.parameters():
            param.requires_grad = False

        # TODO: create the LSTM

        self.lstm = nn.LSTM(
            input_size=resnet_output,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=bidirectional,
            dropout=lstm_dropout,
            batch_first=True
        )

        # TODO: create the FC to generate the translation (3) and rotation (4)
        
        # Create the FC to generate the translation (3) and rotation (4)
        lstm_output_size = hidden_size * 2 if bidirectional else hidden_size
        
        self.fc1 = nn.Linear(lstm_output_size, lstm_output_size//2)
        self.fc2 = nn.Linear(lstm_output_size//2, lstm_output_size//4)
        self.fc3 = nn.Linear(lstm_output_size//4, 7)
        

    def resnet_transforms(self) -> Callable:
        return weights.DEFAULT.transforms(antialias=True)

    def forward(self, x: torch.TensorType) -> torch.TensorType:

        # CNN feature extraction
        batch_size, seq_length, channels, height, width = x.size()
        features = x.view(batch_size * seq_length, channels, height, width)

        with torch.no_grad():
            features = self.cnn_model(features)
            
        features = features.view(batch_size, seq_length, -1)
            
        #print('s-1',features.shape)

        # TODO: use the LSTM
        lstm_out, _ = self.lstm(features)
        
        #print('s0',lstm_out.shape)
        
        x = self.fc1(lstm_out)
        x = torch.relu(x)
        x = self.fc2(x)
        x = torch.relu(x)
        translation_rotation = self.fc3(x)



        # TODO: Get the output of the last time step
        #print('s1',translation_rotation.shape)
        lstm_out_last = translation_rotation[:,-1,:]
        #print('s2',lstm_out_last.shape)

        

        return lstm_out_last
