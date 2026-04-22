import argparse
import json
import os
import time

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


LATENT_DIM   = 100
CHANNELS_IMG = 3
FEATURES_G   = 64
FEATURES_D   = 64
CRITIC_ITER  = 5
WEIGHT_CLIP  = 0.01
LEARNING_RATE = 5e-5
BATCH_SIZE   = 64
IMAGE_SIZE   = 32

CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]


class Generator(nn.Module):
    """
    Latent vector → CIFAR image (3 × 32 × 32)
    Starts at 4×4 feature map, upsamples 3× to reach 32×32.
    """
    def __init__(self, latent_dim: int = LATENT_DIM, features: int = FEATURES_G):
        super().__init__()
        self.net = nn.Sequential(
            # latent_dim × 1 × 1  →  features*8 × 4 × 4
            self._block(latent_dim,     features * 8, 4, 1, 0),
            # → features*4 × 8 × 8
            self._block(features * 8,  features * 4, 4, 2, 1),
            # → features*2 × 16 × 16
            self._block(features * 4,  features * 2, 4, 2, 1),
            # → 3 × 32 × 32
            nn.ConvTranspose2d(features * 2, CHANNELS_IMG, 4, 2, 1),
            nn.Tanh(),
        )

    @staticmethod
    def _block(in_c, out_c, kernel, stride, pad):
        return nn.Sequential(
            nn.ConvTranspose2d(in_c, out_c, kernel, stride, pad, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
        )

    def forward(self, z):
        return self.net(z)



class Critic(nn.Module):
    """
    CIFAR image → unbounded scalar score.
    No sigmoid — required for Wasserstein training.
    Uses LayerNorm instead of BatchNorm (more stable with WGAN).
    """
    def __init__(self, features: int = FEATURES_D):
        super().__init__()
        self.net = nn.Sequential(
            # 3 × 32 × 32  →  features × 16 × 16
            nn.Conv2d(CHANNELS_IMG, features, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # → features*2 × 8 × 8
            self._block(features,     features * 2, 4, 2, 1),
            # → features*4 × 4 × 4
            self._block(features * 2, features * 4, 4, 2, 1),
            # → 1 × 1 × 1
            nn.Conv2d(features * 4, 1, 4, 1, 0, bias=False),
        )

    @staticmethod
    def _block(in_c, out_c, kernel, stride, pad):
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel, stride, pad, bias=False),
            nn.InstanceNorm2d(out_c, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x):
        return self.net(x).view(-1)


def weights_init(m):
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.normal_(m.weight, 0.0, 0.02)
    elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d)):
        if m.weight is not None:
            nn.init.normal_(m.weight, 1.0, 0.02)
        if m.bias is not None:
            nn.init.zeros_(m.bias)



def get_cifar10_loader(batch_size: int, data_root: str = "./data"):
    transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.5, 0.5, 0.5],
            std =[0.5, 0.5, 0.5],
        ),
    ])
    dataset = torchvision.datasets.CIFAR10(
        root=data_root, train=True, download=True, transform=transform
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=True,
                      num_workers=2, pin_memory=True, drop_last=True)



def train(args):
    device = torch.device("cuda" if args.gpu and torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.sample_dir, exist_ok=True)

    # Models
    gen    = Generator(args.latent_dim).to(device)
    critic = Critic().to(device)
    gen.apply(weights_init)
    critic.apply(weights_init)

    # Optimisers  (RMSprop recommended for WGAN)
    opt_gen    = torch.optim.RMSprop(gen.parameters(),    lr=args.lr)
    opt_critic = torch.optim.RMSprop(critic.parameters(), lr=args.lr)

    loader = get_cifar10_loader(args.batch_size)

    # Fixed noise for consistent sample grids
    fixed_noise = torch.randn(64, args.latent_dim, 1, 1, device=device)

    metrics = []          # list of dicts written to JSON
    global_step = 0

    print(f"Starting WGAN training for {args.epochs} epochs …")

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        critic_losses, gen_losses = [], []

        for batch_idx, (real, _) in enumerate(loader):
            real = real.to(device)
            cur_batch = real.size(0)

            for _ in range(args.critic_iter):
                noise = torch.randn(cur_batch, args.latent_dim, 1, 1, device=device)
                fake  = gen(noise).detach()

                score_real = critic(real)
                score_fake = critic(fake)

                loss_critic = -(score_real.mean() - score_fake.mean())

                opt_critic.zero_grad()
                loss_critic.backward()
                opt_critic.step()

                # Weight clipping  (the original WGAN trick)
                for p in critic.parameters():
                    p.data.clamp_(-args.weight_clip, args.weight_clip)

            noise = torch.randn(cur_batch, args.latent_dim, 1, 1, device=device)
            fake  = gen(noise)
            loss_gen = -critic(fake).mean()

            opt_gen.zero_grad()
            loss_gen.backward()
            opt_gen.step()

            critic_losses.append(loss_critic.item())
            gen_losses.append(loss_gen.item())
            global_step += 1

        mean_c = sum(critic_losses) / len(critic_losses)
        mean_g = sum(gen_losses)    / len(gen_losses)
        w_dist  = -mean_c          # Wasserstein distance estimate

        epoch_time = time.time() - epoch_start
        metrics.append({
            "epoch":          epoch,
            "critic_loss":    round(mean_c, 4),
            "gen_loss":       round(mean_g, 4),
            "wasserstein_dist": round(w_dist, 4),
            "epoch_time_sec": round(epoch_time, 2),
        })

        print(
            f"Epoch [{epoch:>3}/{args.epochs}]  "
            f"Critic: {mean_c:.4f}  Gen: {mean_g:.4f}  "
            f"W-dist: {w_dist:.4f}  ({epoch_time:.1f}s)"
        )

        # Save metrics JSON (the Flask API reads this file)
        with open(args.metrics_file, "w") as f:
            json.dump(metrics, f, indent=2)

        # Save sample grid every N epochs
        if epoch % args.sample_every == 0 or epoch == args.epochs:
            gen.eval()
            with torch.no_grad():
                samples = gen(fixed_noise)
            gen.train()
            grid_path = os.path.join(args.sample_dir, f"epoch_{epoch:04d}.png")
            torchvision.utils.save_image(
                samples, grid_path,
                nrow=8, normalize=True, value_range=(-1, 1)
            )
            print(f"  → Sample grid saved: {grid_path}")

        # Save checkpoint
        if epoch % args.save_every == 0 or epoch == args.epochs:
            ckpt = {
                "epoch":      epoch,
                "gen":        gen.state_dict(),
                "critic":     critic.state_dict(),
                "opt_gen":    opt_gen.state_dict(),
                "opt_critic": opt_critic.state_dict(),
            }
            ckpt_path = os.path.join(args.checkpoint_dir, f"wgan_epoch_{epoch:04d}.pth")
            torch.save(ckpt, ckpt_path)
            # Also save a "latest" alias for easy loading
            torch.save(ckpt, os.path.join(args.checkpoint_dir, "wgan_latest.pth"))
            print(f"  → Checkpoint saved: {ckpt_path}")

    print("\nTraining complete!")
    print(f"Metrics  → {args.metrics_file}")
    print(f"Samples  → {args.sample_dir}/")
    print(f"Checkpts → {args.checkpoint_dir}/")



def parse_args():
    p = argparse.ArgumentParser(description="WGAN on CIFAR-10 (PyTorch)")
    p.add_argument("--epochs",          type=int,   default=100)
    p.add_argument("--batch-size",      type=int,   default=BATCH_SIZE)
    p.add_argument("--lr",              type=float, default=LEARNING_RATE)
    p.add_argument("--latent-dim",      type=int,   default=LATENT_DIM)
    p.add_argument("--critic-iter",     type=int,   default=CRITIC_ITER)
    p.add_argument("--weight-clip",     type=float, default=WEIGHT_CLIP)
    p.add_argument("--checkpoint-dir",  type=str,   default="./checkpoints")
    p.add_argument("--sample-dir",      type=str,   default="./samples")
    p.add_argument("--metrics-file",    type=str,   default="./training_metrics.json")
    p.add_argument("--sample-every",    type=int,   default=10,
                   help="Save sample grid every N epochs")
    p.add_argument("--save-every",      type=int,   default=25,
                   help="Save checkpoint every N epochs")
    p.add_argument("--gpu",             action="store_true",
                   help="Use CUDA if available")
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
