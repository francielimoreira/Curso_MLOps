import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.model = nn.Sequential(
            nn.Flatten(),            # Flatten a imagem de entrada
            nn.Linear(784, 128),     # Camada linear com 128 neurônios
            nn.ReLU(),               # Função de ativação ReLU
            nn.Dropout(0.2),         # Dropout com uma taxa de dropout de 20%
            nn.Linear(128, 64),      # Segunda camada linear com 64 neurônios
            nn.ReLU(),               # Função de ativação ReLU
            nn.Dropout(0.2),
            nn.Linear(64, 10),       # Camada de saída com 10 neurônios
            nn.LogSoftmax(dim=1)     # Função de ativação softmax para a saída
        )

    def forward(self, x):
        return self.model(x)
    
def save_model(model, filename):
    torch.save(model.state_dict(), filename)

model = MyModel()
save_model(model, 'model.pt')















