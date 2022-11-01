import pandas as pd
import os
import numpy as np
from PIL import Image
from glob import glob
from tqdm import tqdm
import torch as th
import torchvision as tv 
import torchvision.transforms as T
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
import json

# Pretrained ResNeXT model https://pytorch.org/vision/stable/models.html 
inet_model = tv.models.resnext101_32x8d(pretrained=True) 
inet_model.eval()
n_embed = inet_model.fc.in_features # no. of outputs from resnext, no. of inputs for NN linear layer later
inet_model.fc = nn.Identity()

def preprocess_image(model): # uses pretrained model and image
  preprocess = T.Compose([
    T.Resize(256),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
  
  tensor_dict = {}
  counter = 0

  for path in tqdm(glob('food/*.jpg')):
    image = Image.open(path)
    input_tensor = preprocess(image)
    input_batch = input_tensor.unsqueeze(0)

    with th.no_grad():
      emb = model(input_batch)[0]
    img_name = os.path.splitext(os.path.basename(path))[0]
    tensor_dict[img_name] = emb

  return tensor_dict

tensor_dict = preprocess_image(inet_model) # generate image embeddings saved as tensors
tensor_df = pd.DataFrame.from_dict(tensor_dict)

# Load train triplet datasets
train_path = "train_triplets.txt"

train_triplets = []
with open(train_path) as f:
  for line in f:
    names = line.split()
    a, p, n = names[0], names[1], names[2]
    train_triplets.append((a,p,n))

val_n = 3500
valid_triplets = train_triplets[:val_n]
train_triplets = train_triplets[val_n:]

# Make our own class dataset - adapted from https://androidkt.com/load-pandas-dataframe-using-dataset-and-dataloader-in-pytorch/ 
class MyDataset():
  def __init__(self, dataframe, triplets):
    self.dataframe = dataframe
    self.triplets = triplets
  
  def __getitem__(self, idx):
      return th.FloatTensor([(self.dataframe[j]) for j in self.triplets[idx]])
  
  def __len__(self):
    return len(self.triplets) 

train_data = MyDataset(tensor_df, train_triplets) 
valid_data = MyDataset(tensor_df, valid_triplets) 

# Model and Optimiser + dropout to avoid overfitting
model = nn.Sequential(
    nn.Linear(n_embed, 512),
    nn.ReLU(inplace=True),
    nn.Dropout(0.5),
    nn.Linear(512,256),
    nn.ReLU(inplace=True),
    nn.Linear(256,128),
    nn.ReLU(inplace=True),
    nn.Linear(128,128)) # this will be the input size for our triplet loss

opt = th.optim.Adam(model.parameters()) # Adam optimser
scheduler = th.optim.lr_scheduler.ReduceLROnPlateau(opt, 'min', patience = 5) # reduce learning rate if val_error not improved after 5 epochs

train_loader = DataLoader(train_data, batch_size = 128, shuffle = True, pin_memory = True) 
valid_loader = DataLoader(valid_data, batch_size = 128)

model.eval()
error = 0
total = 0
for batch in valid_loader:
  x = batch.view((-1,) + batch.shape[2:])
  y = model(x).view(batch.shape[:2] +(-1,))
  a_to_p = th.linalg.norm(y[:,0,:]-y[:,1,:], dim=1) 
  a_to_n = th.linalg.norm(y[:,0,:]-y[:,2,:], dim=1)
  total += a_to_p.shape[0] 
  error += (a_to_p >= a_to_n).sum().item()
error = error/total
print('Initialised error', error)
best_error = error
th.save(model.state_dict(), 'model.pt')

# Training
for epoch in range(50):
    model.train()
    for batch in train_loader:
        x = batch.view((-1,) + batch.shape[2:])
        y = model(x).view(batch.shape[:2] +(-1,))
        loss = F.triplet_margin_loss(y[:,0,:], y[:,1,:], y[:,2,:])
        opt.zero_grad()
        loss.backward()
        opt.step()

    # Validation

    model.eval()
    error = 0
    total = 0
    for batch in valid_loader:
      x = batch.view((-1,) + batch.shape[2:])
      y = model(x).view(batch.shape[:2] +(-1,))
      a_to_p = th.linalg.norm(y[:,0,:]-y[:,1,:], dim=1) 
      a_to_n = th.linalg.norm(y[:,0,:]-y[:,2,:], dim=1)
      total += a_to_p.shape[0] 
      error += (a_to_p >= a_to_n).sum().item()
    error = error/total 
    print('Epoch: ', epoch, 'Error: ', error)
    if error < best_error:
        best_error = error
        th.save(model.state_dict(), 'model.pt')
    scheduler.step(error)

# Load test triplet dataset 
test_path = "test_triplets.txt"

test_triplets = []
with open(test_path) as f:
  for line in f:
    names = line.split()
    a, p, n = names[0], names[1], names[2]
    test_triplets.append((a,p,n))

test_data = MyDataset(tensor_df, test_triplets) 
test_loader = DataLoader(test_data, batch_size = 1) 

# Predict on test triplets!
counter = 0
predictions = [] # list of predictions 
for d in test_loader:
    x = d.view((-1,) + d.shape[2:])
    with th.no_grad():
        y = model(x).view(d.shape[:2] +(-1,))
    a_to_1 = th.linalg.norm(y[:,0,:]-y[:,1,:], dim=1) 
    a_to_2 = th.linalg.norm(y[:,0,:]-y[:,2,:], dim=1)
    if a_to_1 < a_to_2: # 1 if anchor closer to 2nd image, 0  closer to 3rd
        predictions.append(1)
    else:
        predictions.append(0)
    
    counter+=1
    if counter%100 ==0:
        print(counter)
