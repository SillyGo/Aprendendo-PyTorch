import numpy as np
import pandas as pd
import torch
import torch.nn as nn

data_in = pd.read_csv('trip2_potholes.csv')
data_out= pd.read_csv('trip2_sensors.csv')

sensor_data = data_out.sort_values('timestamp', ascending=True)
acc = sensor_data['accelerometerY'] #acho que isso é suficiente para pegar as informações do acelerometro y, testarei isso agora.
time_stamp = [x for x in range(1987)]
pothole_data = data_in.sort_values('timestamp', ascending=True)['timestamp'].values # Extract the 'timestamp' column and get the values as a NumPy array
for i in range(len(pothole_data)): # Iterate through the array
  pothole_data[i] = int((pothole_data[i] - 1493001885.2)*5)

labels = []
for i in range(len(time_stamp)):
  if (time_stamp[i] in pothole_data):
    labels.append(1)
  else:
    labels.append(0)


labels = np.array(labels)
acc = np.array(acc)

def create_sequences(data, seq_length):
    xs = []
    classes = []
    for i in range(seq_length):
      temp = []
      for j in range(int(len(data)/seq_length)):
        temp.append(data[j + i*int(len(data)/seq_length)])
      xs.append(temp)
    return xs

def label_image(labels, seq_lenght):
  out = []
  for i in range(seq_lenght):
    for j in range(int(len(labels)/seq_lenght)):
      if (labels[j + i*int(len(labels)/seq_lenght)] == 1):
        out.append(1)
        break
      elif j == int(len(labels)/seq_lenght) - 1:
        out.append(0)
  return out

seq = 20
xs = create_sequences(acc, seq)
ys = label_image(labels, seq)

print("---")
counter = 0
for i in range(len(ys)):
  if ys[i] == 1:
    counter = counter + 1
    print(i)
print("---")
print(f'{100*(counter/seq)}%')
print("---")
trainX = torch.tensor(xs, dtype=torch.float32)
trainY = torch.tensor(ys, dtype=torch.float32)

print(len(trainX))
print(trainY.shape)

training_set = torch.utils.data.TensorDataset(trainX, trainY)
print(training_set)

class NeuralNet(nn.Module): 
    def __init__(self, in_size, hide_size, out_size): #define o geral da rede, ou seja, sua arquitetura e funções de ativação
        super(NeuralNet, self).__init__()
        self.l1 = nn.Linear(in_size, hide_size) #layer 1 
        self.relu = nn.ReLU()                   #funçao de ativação
        self.l2 = nn.Linear(hide_size, out_size)#layer 2

    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        return out 
    
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

in_size = 99
hide_size = 128
num_classes = 2
num_epochs = 100000
batch_size = 20
lrate = 0.001

train_loader = torch.utils.data.DataLoader(dataset=training_set, batch_size=batch_size, shuffle=True)
print(train_loader)

model = NeuralNet(in_size=in_size, hide_size=hide_size, out_size=num_classes)
criterion = nn.CrossEntropyLoss() #novamente, aplica o softmax por nos
optimizer = torch.optim.SGD(model.parameters(), lr=lrate)

for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        #devemos agora fazer um reshape, o t100amanho da imagem, como visto em um print anterior, é [100,1,28,28], queremos transforma-lo em um formato de 100, 784
        images = images.to(device)
        labels = labels.to(device)

        #forward:
        output = model(images)
        loss = criterion(output, labels.long()) # Convert labels to Long type

        #backward:

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if epoch % 1000 == 0:
        print(f'epoch: {epoch}, / loss: {loss.item()}')