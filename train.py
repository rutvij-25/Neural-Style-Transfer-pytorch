import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
import torchvision.transforms as transforms
from torchvision import models
from torchvision.utils import save_image

from models import VGG
import argparse

argparser = argparse.ArgumentParser()

argparser.add_argument('--image_size',type=int,default=256)
argparser.add_argument('--steps',type=int,default=6000)
argparser.add_argument('--learning_rate',type=float,default=0.001)
argparser.add_argument('--alpha',type=float,default=1)
argparser.add_argument('--beta',type=float,default=0.01)
argparser.add_argument('--content_root',type=str,default='images/content.jpg')
argparser.add_argument('--style_root',type=str,default='images/style.jpg')
argparser.add_argument('--gen_root',type=str,default='images/generated.jpg')

args = argparser.parse_args()


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

image_size = args.image_size

loader = transforms.Compose(
    [
     transforms.Resize((image_size,image_size)),
     transforms.ToTensor(),
    ]
)

def load_image(root):
  image = Image.open(root)
  image =  loader(image).unsqueeze(0)
  return image.to(device)

content = load_image(args.content_root)
style = load_image(args.style_root)

generated = content.clone().requires_grad_(True)

STEPS = args.steps
lr = args.learning_rate
alpha = args.alpha
beta = args.beta

model = VGG().to(device)
optimizer = optim.Adam([generated],lr)

for step in range(STEPS):

  generated_features = model(generated)
  content_features = model(content)
  style_features = model(style)

  content_loss = style_loss = 0

  for gen,con,st in zip(generated_features,content_features,style_features):

    batch_size,channel,height,width = gen.shape

    content_loss += torch.mean((gen - con) ** 2)

    #Compute gram matrix
    G = gen.view(channel,height*width).mm(gen.view(channel,height*width).t())
    A = st.view(channel,height*width).mm(st.view(channel,height*width).t())

    style_loss += torch.mean((G - A)**2)
    
  total_loss = alpha * content_loss + beta * style_loss
  optimizer.zero_grad()
  total_loss.backward()
  optimizer.step()
  
  if(step%50==0):
    print(f'STEP:{step} LOSS:{total_loss.item()}')
  save_image(generated,args.gen_root)
