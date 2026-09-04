import tqdm
import math
import torch
from torch import nn
import matplotlib.pyplot as plt
from dataloader import getDataLoader
from vae import VAE
from train import VAEConfig, train

device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    device = 'mps'
print(device)

class Block(nn.Module):
    def __init__(self, channels=512):
        super().__init__()
        self.ln = nn.LayerNorm(channels)
        self.ff = nn.Linear(channels, channels)
        self.act = nn.GELU()

    def forward(self, x):
        return x + self.act(self.ff(self.ln(x)))

class MLP(nn.Module):
    def __init__(self, channels_data=784, layers=5, channels=512, channels_t=512, p_drop = 0.2):
        super().__init__()
        self.channels_t = channels_t
        self.p_drop = p_drop
        
        self.in_projection = nn.Linear(channels_data, channels)
        self.label_embed = nn.Embedding(16, channels) # 10 + 1 = 11 total digit labels. 16 for faster kernals
        self.t_projection = nn.Linear(channels_t, channels)
        
        self.ln = nn.LayerNorm(channels)
        self.blocks = nn.Sequential(*[
            Block(channels) for _ in range(layers)
        ])
        self.out_projection = nn.Linear(channels, channels_data)

    def gen_t_embedding(self, t, w_min = 1.0, w_max=10000.0):
        dimension = self.channels_t
        half_dim = dimension // 2
        
        w = w_min * ((w_max / w_min) ** ((torch.linspace(start=1, end=half_dim, steps=half_dim, device=t.device) - 1) / (half_dim - 1)))
        args = 2 * math.pi * t * w.unsqueeze(0) # 64 x 256
        
        emb = math.sqrt(2.0 / dimension) * torch.cat([torch.cos(args), torch.sin(args)], dim=1) # Batch x t_channels
        if dimension % 2 == 1:  # zero pad
            emb = nn.functional.pad(emb, (0, 1), mode='constant')
        return emb
    
    def dropout_classification(self, y):
        drop = torch.rand(y.shape, device=device) < self.p_drop
        y = y.masked_fill(drop, 10) # 10 is the null classification
        
        return y

    def forward(self, x, t, y):
        x = self.in_projection(x)
        y = self.label_embed(self.dropout_classification(y))
        
        t = self.gen_t_embedding(t)
        t = self.t_projection(t)
        
        x = self.ln(x + t + y)
        x = self.blocks(x)
        x = self.out_projection(x)
        return x


batch_size = 256
train_loader = getDataLoader('test', batch_size=batch_size)

# -----------------------------------------------
vae = VAE(VAEConfig).to(device=device)

import os
ckpt_path = "vaemnist_30001.pt"
if os.path.exists():
    vae.load_state_dict(torch.load(ckpt_path, weights_only=True)['model'])
else:
    print("pretrained vae not found, training")
    train(vae)
    
vae.eval()
for child in vae.children():
    for params in child.parameters():
        params.requires_grad = False

model = MLP(layers=4, channels_data=VAEConfig.latent_channels, channels=512).to(device=device)
optim = torch.optim.AdamW(model.parameters(), lr=3e-3, fused=True)
        
# -----------------------------------------------

model.train().requires_grad_(True)
for epoch in range(10):
    pbar = tqdm.tqdm(train_loader)
    for batch_idx, (images, labels) in enumerate(pbar):
        z = images.to(device)
        y = labels.to(device)
            
        with torch.no_grad():
            z, _, _ = vae.encode(z)        
        epsilon = torch.randn_like(z, device=device)
        
        t = torch.rand(z.size(0), 1, device=device)
        xt = t * z + (1 - t) * epsilon
        loss = ((model(xt, t, y) - (z - epsilon))**2).mean()
        
        optim.zero_grad(set_to_none=True)
        loss.backward()
        optim.step()
        
        pbar.set_postfix(loss=loss.item())
model.eval()

# -----------------------------------------------
# Sampling 
from torchvision.utils import save_image
sampled_images = 16
steps = 100
w = 4.0

model.eval()
@torch.no_grad()
def create_classification(id):
    return torch.full_like(torch.ones(sampled_images), fill_value=id, dtype=torch.int32, device=device)

@torch.no_grad()
def sample_image(id):
    outputs = []
    X_0 = torch.randn(sampled_images, VAEConfig.latent_channels, device=device)
    
    null_classification = create_classification(10)
    for _, t in enumerate(torch.linspace(0, 1, steps, device=device), start=1):
        outputs.append(vae.decoder(X_0))
        
        X_0 = X_0 + (1 / steps) * ((1 - w) * model(X_0, t.view(1,1), null_classification) + w * model(X_0, t.view(1,1), create_classification(id))) 
    return outputs # X_0, ..., X_1

@torch.no_grad()
def show_image(batch_idx, number):
    outputs = sample_image(number)
    X_i = outputs[steps - 1][batch_idx].view(28, 28).cpu()
    plt.figure(figsize=(6, 6))
    plt.imshow(X_i.squeeze(), cmap="gray", vmin=0, vmax=1)
    plt.show()

save_image(sample_image(10)[steps - 1].view(sampled_images, 1, 28, 28), "sampled.png", nrow=4)
