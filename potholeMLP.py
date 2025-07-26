import numpy as np
import pandas as pd
import torch
import torch.nn as nn

#TRIP 2:

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


labels1 = np.array(labels)
acc1 = np.array(acc)

#TRIP 3:

data_in = pd.read_csv('trip3_potholes.csv')
data_out= pd.read_csv('trip3_sensors.csv')

sensor_data = data_out.sort_values('timestamp', ascending=True)
acc = sensor_data['accelerometerY'] #acho que isso é suficiente para pegar as informações do acelerometro y, testarei isso agora.

time_stamp = [x for x in range(2210)]
pothole_data = data_in.sort_values('timestamp', ascending=True)['timestamp'].values # Extract the 'timestamp' column and get the values as a NumPy array
for i in range(len(pothole_data)): # Iterate through the array
  pothole_data[i] = int((pothole_data[i] - 1493002313.5)*5)

labels = []
for i in range(len(time_stamp)):
  if (time_stamp[i] in pothole_data):
    labels.append(1)
  else:
    labels.append(0)

labels2 = np.array(labels)
acc2 = np.array(acc)

#TRIP 4:

data_in = pd.read_csv('trip4_potholes.csv')
data_out= pd.read_csv('trip4_sensors.csv')

sensor_data = data_out.sort_values('timestamp', ascending=True)
acc = sensor_data['accelerometerY'] #acho que isso é suficiente para pegar as informações do acelerometro y, testarei isso agora.

time_stamp = [x for x in range(len(acc))]
pothole_data = data_in.sort_values('timestamp', ascending=True)['timestamp'].values # Extract the 'timestamp' column and get the values as a NumPy array
for i in range(len(pothole_data)): # Iterate through the array
  pothole_data[i] = int((pothole_data[i] - 1493002780.6)*5)

labels = []
for i in range(len(time_stamp)):
  if (time_stamp[i] in pothole_data):
    labels.append(1)
  else:
    labels.append(0)

labels3 = np.array(labels)
acc3 = np.array(acc)

#COLANDO AS DIFERENTES TRIPS EM UM SÓ:

labels = []
acc = []
for i in range(len(labels1)):
  labels.append(labels1[i])
  acc.append(acc1[i])

for i in range(len(labels2)):
  labels.append(labels2[i])
  acc.append(acc2[i])

for i in range(len(labels3)):
  labels.append(labels3[i])
  acc.append(acc3[i])

acc = np.array(acc)
labels = np.array(labels)

print(len(acc))
print(len(labels))

#CRIANDO AS IMAGENS:

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


seq = 50
xs = create_sequences(acc, seq)
ys = label_image(labels, seq)

print(len(xs[0]))

counter = 0
for i in range(len(ys)):
  if ys[i] == 1:
    counter = counter + 1
print("---")
print(f'{100*(counter/len(ys))}%')
print("---")

trainX = torch.tensor(xs, dtype=torch.float32)
trainY = torch.tensor(ys, dtype=torch.float32)

training_set = torch.utils.data.TensorDataset(trainX, trainY)

#criando o modelo

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

in_size = 115
hide_size = 128
num_classes = 2
num_epochs = 20000
batch_size = 20
lrate = 0.001

train_loader = torch.utils.data.DataLoader(dataset=training_set, batch_size=batch_size, shuffle=True)
print(train_loader)

model = NeuralNet(in_size=in_size, hide_size=hide_size, out_size=num_classes)
criterion = nn.CrossEntropyLoss() #novamente, aplica o softmax por nos
optimizer = torch.optim.SGD(model.parameters(), lr=lrate)


#treinamento;

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
    elif epoch == num_epochs - 1:
      print(f'epoch: {epoch}, / loss: {loss.item()}')


#verif

data_in = pd.read_csv('trip4_potholes.csv')
data_out= pd.read_csv('trip4_sensors.csv')

sensor_data = data_out.sort_values('timestamp', ascending=True)
acc = sensor_data['accelerometerY'] #acho que isso é suficiente para pegar as informações do acelerometro y, testarei isso agora.

time_stamp = [x for x in range(len(acc))]
pothole_data = data_in.sort_values('timestamp', ascending=True)['timestamp'].values # Extract the 'timestamp' column and get the values as a NumPy array
for i in range(len(pothole_data)): # Iterate through the array
  pothole_data[i] = int((pothole_data[i] - 1493003562.6)*5)

labels = []
for i in range(len(time_stamp)):
  if (time_stamp[i] in pothole_data):
    labels.append(1)
  else:
    labels.append(0)

labels4 = np.array(labels)
acc4 = np.array(acc)


test_set = torch.utils.data.TensorDataset(trainX, trainY)

test_loader  = torch.utils.data.DataLoader(dataset=test_set, 
                                           batch_size=batch_size, 
                                           shuffle=False)

with torch.no_grad():
    n_correct = 0
    n_samples = 0
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)

        #torch.max() retorna o valor e o index, estamos interessados somente no index, de modo que salvamos o outro numa variavel _, usada para indicar que não usaremos o valor.
        _, predcits = torch.max(outputs,1)
        n_samples += labels.shape[0] #numero de samples para uma batch, deve retornar 100
        n_correct += (predcits == labels).sum().item()

    acc = 100.0*(n_correct/n_samples)
    print(f'acc: {acc}%')
