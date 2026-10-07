"""Per-class DCGAN training, checkpointing and image generation."""
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader
from torchvision.utils import make_grid, save_image

from .data import ImageListDataset, gan_transform, list_images
from .models import NZ, Discriminator, Generator, get_device, weights_init


def _infinite(loader):
    while True:
        yield from loader


def diff_augment(x):
    """DiffAugment (Zhao et al., 2020): differentiable color, translation and cutout.

    Applied to both real and fake images before the discriminator so D cannot
    simply memorise a small disease class; gradients still flow back to G.
    """
    b = x.size(0)
    dev = x.device
    # color: brightness, saturation, contrast
    x = x + (torch.rand(b, 1, 1, 1, device=dev) - 0.5)
    mean = x.mean(dim=1, keepdim=True)
    x = (x - mean) * (torch.rand(b, 1, 1, 1, device=dev) * 2) + mean
    mean = x.mean(dim=[1, 2, 3], keepdim=True)
    x = (x - mean) * (torch.rand(b, 1, 1, 1, device=dev) + 0.5) + mean
    # translation by up to 1/8 of the image, zero padded
    h, w = x.shape[2:]
    sh, sw = h // 8, w // 8
    tx = torch.randint(-sh, sh + 1, (b, 1, 1), device=dev)
    ty = torch.randint(-sw, sw + 1, (b, 1, 1), device=dev)
    gb, gx, gy = torch.meshgrid(torch.arange(b, device=dev), torch.arange(h, device=dev),
                                torch.arange(w, device=dev), indexing="ij")
    gx = torch.clamp(gx + tx + 1, 0, h + 1)
    gy = torch.clamp(gy + ty + 1, 0, w + 1)
    padded = F.pad(x, [1, 1, 1, 1])
    x = padded.permute(0, 2, 3, 1).contiguous()[gb, gx, gy].permute(0, 3, 1, 2)
    # cutout: one zeroed square of half the image size
    ch, cw = h // 2, w // 2
    ox = torch.randint(0, h + (1 - ch % 2), (b, 1, 1), device=dev)
    oy = torch.randint(0, w + (1 - cw % 2), (b, 1, 1), device=dev)
    gb, gx, gy = torch.meshgrid(torch.arange(b, device=dev), torch.arange(ch, device=dev),
                                torch.arange(cw, device=dev), indexing="ij")
    gx = torch.clamp(gx + ox - ch // 2, 0, h - 1)
    gy = torch.clamp(gy + oy - cw // 2, 0, w - 1)
    mask = torch.ones(b, h, w, dtype=x.dtype, device=dev)
    mask[gb, gx, gy] = 0
    return x * mask.unsqueeze(1)


def train_dcgan(
    image_dir,
    out_dir,
    iterations=2000,
    batch_size=64,
    lr=2e-4,
    beta1=0.5,
    real_label=0.9,
    width=64,
    augment=True,
    sample_every=200,
    seed=0,
    progress_cb=None,
):
    """Train a DCGAN on all images in `image_dir` (one disease class).

    Saves generator.pt, history.json and sample grids into `out_dir`.
    `progress_cb(step, total, stats)` is called every few steps for UIs.
    """
    torch.manual_seed(seed)
    device = get_device()
    paths = list_images(image_dir)
    if len(paths) < 2:
        raise ValueError(f"Need at least 2 images to train a GAN, found {len(paths)} in {image_dir}")

    out_dir = Path(out_dir)
    (out_dir / "samples").mkdir(parents=True, exist_ok=True)

    ds = ImageListDataset(paths, gan_transform())
    bs = min(batch_size, len(ds))
    loader = DataLoader(ds, batch_size=bs, shuffle=True, drop_last=True, num_workers=0)
    batches = _infinite(loader)

    netG = Generator(ngf=width).to(device)
    netD = Discriminator(ndf=width).to(device)
    aug = diff_augment if augment else (lambda t: t)
    netG.apply(weights_init)
    netD.apply(weights_init)

    criterion = nn.BCELoss()
    optD = torch.optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
    optG = torch.optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))
    fixed_noise = torch.randn(64, NZ, 1, 1, device=device)

    history = {"step": [], "loss_D": [], "loss_G": [], "D_x": [], "D_G_z": []}
    start = time.time()

    for step in range(1, iterations + 1):
        real = next(batches).to(device)
        b = real.size(0)

        # (1) Update D: maximize log D(x) + log(1 - D(G(z)))
        netD.zero_grad()
        out_real = netD(aug(real))
        # One-sided label smoothing (real=0.9) keeps D from overpowering G on tiny datasets
        loss_real = criterion(out_real, torch.full((b,), real_label, device=device))
        noise = torch.randn(b, NZ, 1, 1, device=device)
        fake = netG(noise)
        out_fake = netD(aug(fake.detach()))
        loss_fake = criterion(out_fake, torch.zeros(b, device=device))
        loss_D = loss_real + loss_fake
        loss_D.backward()
        optD.step()

        # (2) Update G: maximize log D(G(z))
        netG.zero_grad()
        out = netD(aug(fake))
        loss_G = criterion(out, torch.ones(b, device=device))
        loss_G.backward()
        optG.step()

        if step % 10 == 0 or step == iterations:
            history["step"].append(step)
            history["loss_D"].append(loss_D.item())
            history["loss_G"].append(loss_G.item())
            history["D_x"].append(out_real.mean().item())
            history["D_G_z"].append(out.mean().item())
            if progress_cb:
                progress_cb(step, iterations, {k: v[-1] for k, v in history.items()})

        if step % sample_every == 0 or step == iterations:
            netG.eval()
            with torch.no_grad():
                grid = netG(fixed_noise).cpu()
            netG.train()
            save_image(grid, out_dir / "samples" / f"step_{step:06d}.png", nrow=8, normalize=True, value_range=(-1, 1))

    torch.save({"state_dict": netG.state_dict(), "nz": NZ, "width": width, "class_dir": str(image_dir),
                "n_real": len(paths), "iterations": iterations}, out_dir / "generator.pt")
    history["seconds"] = time.time() - start
    (out_dir / "history.json").write_text(json.dumps(history))
    return history


def load_generator(ckpt_path, device=None):
    device = device or get_device()
    ckpt = torch.load(ckpt_path, map_location=device)
    netG = Generator(nz=ckpt.get("nz", NZ), ngf=ckpt.get("width", 64)).to(device)
    netG.load_state_dict(ckpt["state_dict"])
    netG.eval()
    return netG


@torch.no_grad()
def generate_images(netG, n, seed=None, batch=64):
    """Return n PIL images (64x64) sampled from the generator."""
    device = next(netG.parameters()).device
    gen = torch.Generator(device=device)
    if seed is not None:
        gen.manual_seed(seed)
    else:
        gen.seed()
    imgs = []
    remaining = n
    while remaining > 0:
        k = min(batch, remaining)
        z = torch.randn(k, netG.nz, 1, 1, device=device, generator=gen)
        out = (netG(z).cpu() + 1) / 2  # [-1,1] -> [0,1]
        for t in out.clamp(0, 1):
            arr = (t.permute(1, 2, 0).numpy() * 255).round().astype("uint8")
            imgs.append(Image.fromarray(arr))
        remaining -= k
    return imgs


def image_grid(imgs, nrow=8):
    import torchvision.transforms.functional as TF
    tensors = torch.stack([TF.to_tensor(im) for im in imgs])
    grid = make_grid(tensors, nrow=nrow, padding=2)
    return TF.to_pil_image(grid)
