import torch.nn as nn

class CovidPredictorLSTM(nn.Module):
    def __init__(self, n_features, hidden_size, n_layers):
        super().__init__()
        self.hidden_size = hidden_size
        self.n_layers = n_layers

        self.lstm = nn.LSTM(
            input_size = n_features,
            hidden_size = hidden_size,
            num_layers = n_layers,
            batch_first = True
        )
        self.fc = nn.Linear(hidden_size, 1) 

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :]) 
        return out