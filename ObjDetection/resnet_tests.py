from torchvision.models import resnet50, ResNet50_Weights

weights = ResNet50_Weights.DEFAULT
prep = weights.transforms()     # me dá todas as transformações necessárias para qualquer imagem poder rodar no resnet

print(prep)
