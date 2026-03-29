import torch
import torch.nn as nn
import torchvision
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms

#classe de modelo e device manager

class convnet(nn.Module):
    def __init__(self):
        super(convnet, self).__init__()
        self.pool = nn.MaxPool2d(2,2)           #2x2 pooling size
        self.conv1 = nn.Conv2d(1,6,5)           #1 color channel, kernel size of 5
        self.conv2 = nn.Conv2d(6,16,5)          #drinks from previous conv layer
        self.fc1  =  nn.Linear(16*16, 120)      #drinks from previous conv layer
        self.fc2  =  nn.Linear(120, 84)         #drinks from previous fc layer
        self.fc3  =  nn.Linear(84, 10)          #drinks from previous fc layer
    
    def forward(self, x):
        x = F.relu(self.pool(self.conv1(x)))
        #print(x.shape)
        x = F.relu(self.pool(self.conv2(x)))
        #print(x.shape)
        x = x.view(-1,16*16)
        #print(x.shape)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x
    
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#alguns parametros:

batch_size = 5
num_epochs = 10
learning_rate = 0.001

#importando o dataset
training_dataset = torchvision.datasets.MNIST(root='./data', 
                                              train=True,
                                              transform=transforms.ToTensor(), 
                                              download=True)
test_dataset = torchvision.datasets.MNIST(root='./data', 
                                          train=False,transform=transforms.ToTensor())
train_loader = torch.utils.data.DataLoader(dataset=training_dataset, 
                                           batch_size=batch_size, 
                                           shuffle=True) 
test_loader  = torch.utils.data.DataLoader(dataset=test_dataset, 
                                           batch_size=batch_size, 
                                           shuffle=False)

#criando o modelo, definindo a loss function, e definindo o otimizador
model = convnet().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate)

n_total_steps = len(train_loader)


for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)
        #forward:
        output = model(images)
        loss = criterion(output, labels)
        #backward:
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (i + 1) % 100 == 0:
            print(f'epoch: {i+1}, / loss: {loss.item()}')

print("treinamento concluído! :D")


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
    print(f'acc: {acc}')