import torch
import torch.nn as nn

#para o proposito do problema (me iniciar em LSTM + torch),
#escolherei um problema simples, que nesse caso será o 
#treinamento de um modelo RNN para funções senoidais. Sim,
#isso também pode ser usado em um MLP, mas estou usando em 
#LSTM para me familiarizar com o esquema, de modo a conseguir
#implementar coisas mais complexas com uma maior facilidade.

#ETAPA 1: gerando o modelo

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
class LSTMmod(nn.Module):
    def __init__(self, in_size, h_size, layer_size, out_size):
        super(LSTMmod, self).__init__()
        self.h_size = h_size
        self.layer_size = layer_size
        self.lstm1 = nn.LSTM(in_size, h_size, layer_size, batch_first=True)
        self.fc1 = nn.Linear(h_size,out_size)

    def forward(self, x, h0=None, c0=None):
        if (h0 == None or c0 == None):
            h0 = torch.zeros(self.layer_size, x.size(0), self.h_size).to(x.device)
            c0 = torch.zeros(self.layer_size, x.size(0), self.h_size).to(x.device)
        out, (hn, cn) = self.lstm1(x,(h0,c0))
        out = self.fc1(out[:,-1,:]) #ht
        return out, hn, cn

model = LSTMmod(1,100,1,1)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

#ETAPA 2: GERAÇÃO DE DATASET:   #essa parte foi tirada do geeks 4 sla quanto, para acelerar o processo de estudo.

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)
torch.manual_seed(0)

t = np.linspace(0, 100, 1000)
data = np.sin(t)

def create_sequences(data, seq_length):
    xs = []
    ys = []
    for i in range(len(data)-seq_length):
        x = data[i:(i+seq_length)]
        y = data[i+seq_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

seq_length = 10
X, y = create_sequences(data, seq_length)

trainX = torch.tensor(X[:, :, None], dtype=torch.float32)
trainY = torch.tensor(y[:, None], dtype=torch.float32)

#ETAPA 3: TREINAMENTO:
num_epochs = 100
h0, c0 = None, None

for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()

        output, h0, c0 = model(trainX, h0, c0)
        loss = criterion(output, trainY)
        loss.backward()
        optimizer.step()

        h0, c0 = h0.detach(), c0.detach()

        if (epoch + 1) % 1 == 0:
            print(f'epoch: {epoch+1}, / loss: {loss.item()}')

print("treinamento concluído! :D")

