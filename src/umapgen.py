import torch
import numpy as np
import matplotlib.pyplot as plt
import umap

def generate_umap(model, dataloader):
    latents = []
    digits = []

    device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(dataloader):
            images = images.to(device)
            mu = model.mulayer(model.encode(images))
            
            latents.append(mu.cpu().numpy())
            digits.append(labels.cpu().numpy())

    latents = np.concatenate(latents, axis=0)
    digits = np.concatenate(digits, axis=0)

    reducer = umap.UMAP()
    reducer.fit_transform(latents)

    fig, ax = plt.subplots(figsize=(10, 8))

    scatter = ax.scatter(
        x=reducer.embedding_[:, 0],
        y=reducer.embedding_[:, 1],
        c=digits,
        cmap='tab10',
        alpha=0.7,
        s=15,
        edgecolors='none'
    )

    ax.set_title('UMAP Projection of VAE Latent Space', fontsize=14)
    ax.set_xlabel('Dimension 1', fontsize=12)
    ax.set_ylabel('Dimension 2', fontsize=12)

    ax.legend(
        *scatter.legend_elements(),
        bbox_to_anchor=(1.05, 1),
        loc='upper left',
        title='Classes'
    )

    plt.tight_layout()
    plt.savefig('UMAP-VAE.png', dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()