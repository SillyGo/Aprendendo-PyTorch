import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class LinearRegressionModel(nn.Module): #pai: nn.module, pegaremos nn.linear desse carinha
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)      

    def forward(self, x):   #define o forward_pass;
        return self.linear(x).squeeze(1)    #eu quero que você retorne um valor 1d

X = np.array([x for x in range(0,100)])
X = X.reshape(-1,1)
y = 46 + X.flatten()*2
noise = np.random.normal(0, 10, len(X)) 
y = y + noise

x_mean, x_std = X.mean(), X.std()   #mean e standart deviation
X_norm = (X - x_mean) / x_std       #normalização
X_tensor = torch.tensor(X_norm, dtype=torch.float32)

y_mean, y_std = y.mean(), y.std()   #mean e standart deviation
y_norm = (y - y_mean) / y_std       #normalização
y_tensor = torch.tensor(y_norm, dtype=torch.float32)

in_feature = 1
out_feature = 1

model = LinearRegressionModel(1,1) #construtor da classe
criterion = nn.MSELoss(reduction='mean') #quero que a função de erro seja mean squared error
optimizer = optim.SGD(model.parameters(), lr=0.1)
#logo  que queremos focar em regressão, usamos o SGD, logo que não precisamos de algo 
#com acurácia absurda, e sim alta capacidade de generalização.

num_epochs = 10
for epoch in range(0, num_epochs):
    output = model.forward(X_tensor)
    loss = criterion(output, y_tensor)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(f'loss: {loss.item()}')

new_x = np.array([x for x in range(0,100)])

new_x_normalized = (new_x - x_mean) / x_std

new_x_tensor = torch.tensor(new_x_normalized, dtype=torch.float32)
prediction_normalized = []
model.eval()
with torch.no_grad():
    #prediction_normalized = model(new_x_tensor)
    for i in range(0,100):
        prediction_normalized.append(model(new_x_tensor[i].view(1,-1)))

#pred_denorm = prediction_normalized.item()*y_std + y_mean
pred_denorm = []
for i in range(0,100):
    pred_denorm.append(prediction_normalized[i].item()*y_std + y_mean)

print(pred_denorm)

#CONCLUSÃO: acho que não entendi muito bem a parte da estatística, mas entendi perfeitamente como funciona a arquitetura de
#um modelo pytorch. :D