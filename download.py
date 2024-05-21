import os
import gdown

### Checkpoints ###
os.makedirs("checkpoints", exist_ok=True)

output = 'checkpoints/part.pth'
drive_id = "1tPAtjFj3yb6Af3NjsNBHWhJk--klopk6"
# https://drive.usercontent.google.com/download?id=1tPAtjFj3yb6Af3NjsNBHWhJk--klopk6&export=download&authuser=0&confirm=t&uuid=8446823b-161a-46db-a183-beb3fb3d3f5e&at=APZUnTVj6YvXbrvdkDpadaNOUIYF:1716277040698

url = f'https://drive.google.com/uc?id={drive_id}&confirm=t'
gdown.download(url, output, quiet=False)

output = 'checkpoints/global.pth'
drive_id = "1M1baZqEuYXgucutt50BF-dgwCQtpB5Zg"
# https://drive.usercontent.google.com/download?id=1M1baZqEuYXgucutt50BF-dgwCQtpB5Zg&export=download&authuser=0&confirm=t&uuid=fecf0978-cb0e-4693-8bfb-b8b86700d1d3&at=APZUnTXV-ORtee5magSZobYKZjCd:1716277192446

url = f'https://drive.google.com/uc?id={drive_id}&confirm=t'
gdown.download(url, output, quiet=False)


### Backbones ###
os.makedirs("backbones", exist_ok=True)

output = 'backbones/mae_pretrain_hoi_vit_small.pth'
drive_id = "1zVKlKSpZwNasp7xFaEKP9gTpp0qDFIE3"
# https://drive.usercontent.google.com/download?id=1zVKlKSpZwNasp7xFaEKP9gTpp0qDFIE3&export=download&authuser=0&confirm=t&uuid=46308074-7303-40df-a5be-4b1f18623855&at=APZUnTV40mwiMKN3XIhsBuNx6Awf:1716277153699

url = f'https://drive.google.com/uc?id={drive_id}&confirm=t'
gdown.download(url, output, quiet=False)


### GroundedDINO ###

output = 'grounding_dino/object_souped.pth'
drive_id = "1ZgYztFboBoglYGGp11lvdvwPRhGG8gPa"
# https://drive.usercontent.google.com/download?id=1ZgYztFboBoglYGGp11lvdvwPRhGG8gPa&export=download&authuser=0&confirm=t&uuid=b262280c-d214-42c7-a0fe-a4866623d290&at=APZUnTXuELKrbZJcrgezt9Xn5ccb:1716277117509

url = f'https://drive.google.com/uc?id={drive_id}&confirm=t'
gdown.download(url, output, quiet=False)

output = 'grounding_dino/kitchen_souped.pth'
drive_id = "1xdF8wcpbxAc3RHRFB3uB8xd3LgsvMycV"
# https://drive.usercontent.google.com/download?id=1xdF8wcpbxAc3RHRFB3uB8xd3LgsvMycV&export=download&authuser=0&confirm=t&uuid=225222ed-ad41-4914-a005-bacaeedd5f22&at=APZUnTXaplfbSTIMBAOfUhnHE4ZK:1716277170683

url = f'https://drive.google.com/uc?id={drive_id}&confirm=t'
gdown.download(url, output, quiet=False)