import torch, torchvision, torchaudio
print(torch.__version__)
print(torchvision.__version__)
print(torchaudio.__version__)
print(torch.cuda.is_available())

x = [1,2,3,4,5,6]
y = ['k','j','h','s','f','z']

for i, (xx,yy) in enumerate(zip(x,y)):
    print(i, xx, yy)

