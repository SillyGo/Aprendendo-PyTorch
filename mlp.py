import torch 
import torch.optim.adadelta
import torch.utils.data.dataloader
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

#1. CONFIGURANDO A CLASSE DA REDE.
#depois irei configurar o dataset, mais embaixo, e organizarei ele nas
#devidas subcategorias, para iniciar o treinamento depois.

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
        return out #não apliquei softmax logo que a minha função de perda é categorical crossentropy. Ela vai aplicar o softmax por mim.

#configuração de dispositivo
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

in_size = 784 #28x28, tamanho da imagem
hide_size = 100 #tamanho da hidden layer, eu acho
num_classes = 10 #temos 10 classes diferentes que o programa pode prever.
num_epochs = 2   # numero de epocas, batch_size e learning rate são bem intuitios, assumindo que você sabe o básico
batch_size = 100
lrate = 0.001

#importando o dataset
training_dataset = torchvision.datasets.MNIST(root='./data', 
                                              train=True,
                                              transform=transforms.ToTensor(), 
                                              download=True)

test_dataset = torchvision.datasets.MNIST(root='./data', 
                                          train=False,transform=transforms.ToTensor())   #queremos que seja um tensor, logo, aplicamos um transformador para tensor.

train_loader = torch.utils.data.DataLoader(dataset=training_dataset, 
                                           batch_size=batch_size, 
                                           shuffle=True) #aqui é onde carregamos a informação com a qual verificaremos se o modelo está ok, ele irá ser treinado baseado nos erros que ocorrem nessa fase, por isso estamos dando shuffle.


test_loader  = torch.utils.data.DataLoader(dataset=test_dataset, 
                                           batch_size=batch_size, 
                                           shuffle=False)#aqui é onde carregamos a validation set, com a qual apenas verificaremos nós mesmos se está tudo ok, notadamente, se o modelo não está overfitting.


examples = iter(train_loader)       #examples contêm a training data
samples, labels = next(examples)    #labels contêm... bem... as labels
#print(samples.shape,labels.shape)

model = NeuralNet(in_size=in_size, hide_size=hide_size, out_size=num_classes) #detalhe, o out_size é o numero de classes logo que o comportamento desejado é que a rede retorne uma probabilidade para cada uma das classes listadas, logo, teremos num_classes outputs.

#perda e otimizadores:
criterion = nn.CrossEntropyLoss() #novamente, aplica o softmax por nos
optimizer = torch.optim.Adam(model.parameters(), lr=lrate)

#loop de treinamento:

steps = len(train_loader)

for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        #devemos agora fazer um reshape, o tamanho da imagem, como visto em um print anterior, é [100,1,28,28], queremos transforma-lo em um formato de 100, 784
        images = images.reshape(-1, 28*28).to(device)
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

#testing:
#não queremos computadar todos os gradientes. Usaremos with torch.no_grad():

with torch.no_grad():
    n_correct = 0
    n_samples = 0
    for images, labels in test_loader:
        images, labels = images.reshape(-1,28*28).to(device), labels.to(device)
        outputs = model(images)

        #torch.max() retorna o valor e o index, estamos interessados somente no index, de modo que salvamos o outro numa variavel _, usada para indicar que não usaremos o valor.
        _, predcits = torch.max(outputs,1)
        n_samples += labels.shape[0] #numero de samples para uma batch, deve retornar 100
        n_correct += (predcits == labels).sum().item()

    acc = 100.0*(n_correct/n_samples)
    print(f'acc: {acc}')

#CONCLUSÃO E ANALISE:
#bixo, exatamente que nem no modelo de regressão, eu tou entendendo perfeitamente a parte do treinamento e da arquitetura do modelo, ou seja, as partes que usam mais os conceitos
#de IA em si doque conhecimento puro de programação em python (até as coisas que usam, como class inheritance, eu to pegando dboa), mas oque ta me pegando é a parte da validação. 
#dela, eu entendi só a parte do reshape(), mas não entendi oque é um "index" nas parte do torch.max() (pesquisar), não entendi a parte do .sum(), mas peguei o motivo do .item() só isso
#e a parte com incrementar o n_correct. 