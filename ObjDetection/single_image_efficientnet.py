import torch
from PIL import Image
import torchvision.transforms as transforms
import json
import warnings

warnings.filterwarnings('ignore')   # tirei os warnings pq são chatos

#aqui, definimos qual dispositivo estamos usando. No meu caso é apenas uma CPU.
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f'Using {device} for inference')

#aqui, carregamos o modelo e jogamos ele para o dispositivo (CPU no meu caso)
efficientnet = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_efficientnet_b0', pretrained=True)
efficientnet.eval().to(device)

#aqui, definimos o pre-processamento das imagens, essencial para jogar elas no modelo.
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# IMPORTANTE: por que tem 3 médias e 3 desvios padrões ? Isso é por que a normalização é por canal. Em RGB temos 3 canais, logo, vamos precisar de 3 parâmetros.
# OUTRA COISA: por que a normalização recebe parâmetros ? Isso é como acontece quando usamos o normalizador do sklearn, que encontra a normalização ideal.
# ... O modelo foi treinado com essa normalização, logo, devemos replicar ela.
# ... Enfim, coisas de pre-processamento que, sinceramente, são tão importantes quanto qualquer outra coisa....

img = Image.open('car1.png').convert('RGB')                             #preciso converter para rgb pq png tem 4 canais

#vamos precisar usar o unsqueeze pois o efficient net requer input do formato [N, C, H, W],
# ... Ou seja: batch, channel, height, width. O unsqueeze(0) vai adicionar mais uma dimensão
# ... na posição 0, que é exatamente o que queremos fazer, para adicionar o número de batch.
img_tensor = preprocess(img).unsqueeze(0).to(device)                    

#roda inferência
with torch.no_grad():
    logits = efficientnet(img_tensor)

    #devemos rodar softmax manualmente nos inputs finais, logo que o modelo retorna só os logits (raw scores)
    output = torch.nn.functional.softmax(logits, dim=1)

top_prob, top_idx = torch.topk(output, k=5, dim=1)

with open('LOC_synset_mapping.json', 'r', encoding='utf-8') as f:
    class_names = json.load(f)

print('Top-5 predictions:')
for prob, idx in zip(top_prob[0].tolist(), top_idx[0].tolist()):
    print(f'{idx:4d} | {class_names[idx]} | {prob:.4f}')