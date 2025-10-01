import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
])

train_data = torchvision.datasets.CIFAR10(root='data', train=True, transform=transforms,download=False)
test_data = torchvision.datasets.CIFAR10(root='data', train=False, transform=transforms,download=False)

train_loader = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=True,num_workers=2)
train_loader = torch.utils.data.DataLoader(test_data, batch_size=32, shuffle=True,num_workers=2)

image,label = train_data[0]
#32x32 images

class_names = ['plane','car','bird','cat','deer','dog','frog','horse','ship','truck']

class convnet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3,12,5)
        self.pool = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(12,24,5)
        #reuse pool layer
        self.dense1 = nn.Linear(24*5*5, 124)
        self.dense2 = nn.Linear(124, 64)
        self.dense3 = nn.Linear(64,10)
        
    def forward(self,x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)

        x = torch.flatten(x,1)

        x = F.relu(self.dense1(x))
        x = F.relu(self.dense2(x))
        x = self.dense3(x)
        return x
    
net = convnet()
loss_function = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), 
                      lr=0.001,
                      momentum=0.9)

epoch = 30

for i in range(epoch):
    print(f'epoch: {i}')
    running_loss = 0.0

    for j, data in enumerate(train_loader):
        inputs, labels = data
        optimizer.zero_grad()

        outputs = net(inputs)

        loss = loss_function(outputs,labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f'loss: {running_loss/len(train_loader)}')

