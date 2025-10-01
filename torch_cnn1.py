import torch
import torch.nn as nn
import torch.nn.functional as functional
import torchvision.transforms as transforms
import torch.optim as optim
import torchvision

class convnet(nn.Module):
    def __init__(self):
        super(convnet, self).__init__()
        self.lin_dropout = nn.Dropout(p=0.2)
        self.cnn_dropout = nn.Dropout2d(p=0.2)
        self.conv1 = nn.Conv2d(3,6,5) #aqui, devemos especificar o tamanho dos canais de cor (nesse caso, 3), o tamanho do output fica como escolha do usuário, e o tamanho de kernel, aqui, 5
        self.pool = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(6,16,5) #novamente, o tamanho do kernel fica o mesmo, e o tamanho do output aumenta
        self.fc1 = nn.Linear(16*5*5, 124) #temos 16*5*5 logo que o tamanho final da imagem, depois de todas as transformações impostas pelas camadas de convolução e de maxpooling, será 5x5
        self.fc2 = nn.Linear(124, 64)
        self.fc3 = nn.Linear(64, 10) #nos temos 10 classes diferentes, logo, deveremos ter 10 outputs.

    def forward(self, x):
        x = self.pool(self.cnn_dropout(functional.relu(self.conv1(x))))
        x = self.pool(self.cnn_dropout(functional.relu(self.conv2(x))))
       
        x = x.view(-1, 16*5*5)
        x = self.lin_dropout(functional.relu(self.fc1(x)))
        x = self.lin_dropout(functional.relu(self.fc2(x)))
        x = self.lin_dropout(self.fc3(x))
        
        return x

#estaremos implementando o cifar-10. Isso será um experimento muito interessante.

#configuração de dispositivo
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

num_epochs = 15
batch_size = 4
lrate = 0.001

transform = transforms.Compose(         #carrega os dados e aplica o data augmentation
    [
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=45),
        transforms.RandomVerticalFlip(p=0.05),
        transforms.ColorJitter(brightness=0.5),
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
     ]
)

train_data = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
test_data =  torchvision.datasets.CIFAR10(root='./data', train=False,download=True, transform=transform)

train_loader = torch.utils.data.DataLoader(dataset=train_data, 
                                           batch_size=batch_size, 
                                           shuffle=True)

test_loader  = torch.utils.data.DataLoader(dataset=test_data, 
                                           batch_size=batch_size, 
                                           shuffle=False)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

#model design:

model = convnet().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=lrate)

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