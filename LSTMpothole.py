import numpy as np
import pandas as pd
import torch
import torch.nn as nn

data_in = pd.read_csv('/var/input/trip2_potholes.csv')
data_out= pd.read_csv('/output/trip2_sensors.csv')

sensor_data = data_out.sort_values('timestamp', ascending=True)
ys = sensor_data['accelerometerY'] #acho que isso é suficiente para pegar as informações do acelerometro y, testarei isso agora.

print(ys.size)

time_stamp = [x for x in range(1987)]
pothole_data = data_in.sort_values('timestamp', ascending=True)
pothole_data = np.array(pothole_data)
for i in range(22):
  pothole_data[i] = int((pothole_data[i] - 1493001885.2)*5)

for i in range(22):
  print(pothole_data[i])

labels = []
for i in range(len(time_stamp)):
  if (time_stamp[i] in pothole_data):
    labels.append(1)
  else:
    labels.append(0)

labels = np.array(labels)
sensor_data = np.array(sensor_data)

def create_sequences(data, seq_length):
    xs = []
    ys = []
    for i in range(len(data)-seq_length):
        x = data[i:(i+seq_length)]
        y = data[i+seq_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

_, xs = create_sequences(sensor_data, 10)
_, ys = create_sequences(labels, 10)

trainX = torch.tensor(xs[:, :, None], dtype=torch.float32)
trainY = torch.tensor(ys[:, None], dtype=torch.float32)

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

        if (epoch + 1) % 5 == 0:
            print(f'epoch: {epoch+1}, / loss: {loss.item()}')

print("treinamento concluído! :D")