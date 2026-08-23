import torch
import numpy as np
from dataclasses import dataclass
from vae import VAE
from dataloader import getDataLoader

device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    device = 'mps'
print(device)

# hyperparams
@dataclass
class VAEConfig:
    batch_size: int = 32
    input_dim: int = 784
    hidden_dim: int = 512
    hidden_layers: int = 2
    latent_channels: int = 32
    lr: float = 3e-4
    
# =========================================================================
import os
def save_checkpoint(raw_model, step):
    checkpoint = {
        'model': raw_model.state_dict(),
        'step': step
    }
    torch.save(checkpoint, f"vaemnist_{step}.pt")

def load_checkpoint(ckpt_path, model):
    if not (os.path.exists(ckpt_path)):
        return 1

    checkpoint = torch.load(ckpt_path, weights_only=True)
    model.load_state_dict(checkpoint['model'])

    return checkpoint['step']
# --------------------------------------------------------------------------------
def get_lr(iter):
    return 3e-4 - (6e-9 * iter)

@torch.no_grad()
def test_eval(model, dataloader):
    model.eval()

    total_loss = 0
    batches = 0
    with torch.no_grad():
        for _, (images, _) in enumerate(dataloader):
                with torch.autocast(device_type=device, dtype=torch.bfloat16):
                    images = images.to(device)
                    _, loss, _, _ = model(images, images)

                total_loss += (loss.item())
                batches += 1

    total_loss /= batches
    model.train()
    return total_loss
# =========================================================================

def main():
    model = VAE(VAEConfig()).to(device)
    step = load_checkpoint(ckpt_path="vaemnist_30001.pt", model=model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=model.config.lr)

    train_loader = getDataLoader('train', batch_size=VAEConfig().batch_size)
    test_loader = getDataLoader('test', batch_size=VAEConfig().batch_size)

    model.train()
    while (step < 30000):
        for batch_idx, (images, _) in enumerate(train_loader):
            optimizer.zero_grad(set_to_none=True)
            
            with torch.autocast(device_type=device, dtype=torch.bfloat16):
                images = images.to(device)
                output, loss, recon_loss, kl_loss = model(images, images)

            loss.backward()
            optimizer.step()
            
            
            lr = get_lr(step)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
            
            if (step % 1000 == 0):
                print(f"step: {step} | total loss: {loss.item():.2f} bce {recon_loss.item():.2f} kl {kl_loss.item():.2f} | test: {test_eval(model=model, dataloader=test_loader):.2f} | lr: {lr:.6f}")
                
            step += 1

    save_checkpoint(raw_model=model, step=step)

    # ----------------------------------------------------
    # Sample Results
    # import matplotlib.pyplot as plt
    from torchvision.utils import save_image

    total_samples = 64
    sampled_latents = torch.randn((total_samples, VAEConfig().latent_channels), device=device)

    with torch.no_grad():
        output = torch.sigmoid(model.decode(sampled_latents))
        # plt.imshow(output[0].cpu().detach().reshape(28, 28).numpy(), cmap="gray") or plt.imsave
        save_image(output.view(total_samples, 1, 28, 28), "sampled.png", nrow=8)
        

    # UMAP Results
    from umapgen import generate_umap
    generate_umap(model, test_loader)
    

if __name__ == "__main__":
    main()