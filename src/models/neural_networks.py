import torch
import torch.nn as nn
import torch.nn.functional as F

class CovidPredictorLSTM(nn.Module):
    def __init__(self, n_features, hidden_size, n_layers, dropout = 0.2):
        super(CovidPredictorLSTM, self).__init__()
        self.lstm = nn.LSTM(
            input_size = n_features,
            hidden_size = hidden_size,
            num_layers = n_layers,
            dropout=dropout if n_layers > 1 else 0.0,
            batch_first = True,
            bias = True
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

class GatingNetwork(nn.Module):
    def __init__(self, input_dim, num_experts):
        super().__init__()
        self.gate = nn.Linear(input_dim, num_experts, bias = True)

    def forward(self, x):
        return F.softmax(self.gate(x), dim = -1)

class CovidPredictorPLE(nn.Module):
    def __init__(self, n_features, hidden_size, num_layers = 2, num_experts = 3, dropout = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            dropout=0.0
        )

        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout)
            )
            for _ in range(num_experts)
        ])

        self.gate = GatingNetwork(hidden_size, num_experts)

        self.progressive_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout)
            )
            for _ in range(num_layers)
        ])

        self.fc_out = nn.Linear(hidden_size, 1)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        shared_output = lstm_out[:, -1, :]  

        gate_weights = self.gate(shared_output)  
        expert_outputs = torch.stack([exp(shared_output) for exp in self.experts], dim=1)
        combined = torch.sum(gate_weights.unsqueeze(-1) * expert_outputs, dim=1)

        out = combined
        for layer in self.progressive_layers:
            out = layer(out) + out  

        prediction = self.fc_out(out)
        return prediction
