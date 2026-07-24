import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm import tqdm
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from src.dataset import ArtDataset, transforms
from src.generator import Generator
from src.discriminator import Discriminator


def save_checkpoint(model, optimizer, filename):
    torch.save({
        "state_dict": model.state_dict(),
        "optimizer":  optimizer.state_dict(),
    }, filename)


def load_checkpoint(filename, model, optimizer, lr):
    checkpoint = torch.load(filename, map_location=config.DEVICE)
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    for pg in optimizer.param_groups:
        pg["lr"] = lr


def train_epoch(disc_A, disc_B, gen_A, gen_B, loader,
                opt_disc, opt_gen, l1, mse, d_scaler, g_scaler, epoch):

    loop = tqdm(loader, desc=f"Epoch {epoch}", leave=True)

    for idx, (img_A, img_B) in enumerate(loop):
        img_A = img_A.to(config.DEVICE)
        img_B = img_B.to(config.DEVICE)

        # ── Discriminateurs ──────────────────────────────────────────
        with torch.amp.autocast('cuda'):
            fake_B = gen_B(img_A)   # A → B
            fake_A = gen_A(img_B)   # B → A

            # Disc A
            d_A_real = disc_A(img_A)
            d_A_fake = disc_A(fake_A.detach())
            d_A_loss = mse(d_A_real, torch.ones_like(d_A_real)) + \
                       mse(d_A_fake, torch.zeros_like(d_A_fake))

            # Disc B
            d_B_real = disc_B(img_B)
            d_B_fake = disc_B(fake_B.detach())
            d_B_loss = mse(d_B_real, torch.ones_like(d_B_real)) + \
                       mse(d_B_fake, torch.zeros_like(d_B_fake))

            d_loss = (d_A_loss + d_B_loss) / 2

        opt_disc.zero_grad()
        d_scaler.scale(d_loss).backward()
        d_scaler.step(opt_disc)
        d_scaler.update()

        # ── Générateurs ──────────────────────────────────────────────
        with torch.amp.autocast('cuda'):
            # Adversarial
            loss_gen_B = mse(disc_B(fake_B), torch.ones_like(disc_B(fake_B)))
            loss_gen_A = mse(disc_A(fake_A), torch.ones_like(disc_A(fake_A)))

            # Cycle
            cycle_A = gen_A(fake_B)
            cycle_B = gen_B(fake_A)
            cycle_A_loss = l1(img_A, cycle_A) * config.LAMBDA_CYCLE
            cycle_B_loss = l1(img_B, cycle_B) * config.LAMBDA_CYCLE

            # Identity
            identity_A = gen_A(img_A)
            identity_B = gen_B(img_B)
            identity_A_loss = l1(img_A, identity_A) * config.LAMBDA_IDENTITY
            identity_B_loss = l1(img_B, identity_B) * config.LAMBDA_IDENTITY

            g_loss = (loss_gen_A + loss_gen_B
                      + cycle_A_loss + cycle_B_loss
                      + identity_A_loss + identity_B_loss)

        opt_gen.zero_grad()
        g_scaler.scale(g_loss).backward()
        g_scaler.step(opt_gen)
        g_scaler.update()

        # ── Sauvegarde images tous les 200 batchs ────────────────────
        if idx % 200 == 0:
            save_image(fake_B * 0.5 + 0.5,
                       f"{config.OUTPUT_DIR}/fake_B_epoch{epoch}_batch{idx}.png")
            save_image(fake_A * 0.5 + 0.5,
                       f"{config.OUTPUT_DIR}/fake_A_epoch{epoch}_batch{idx}.png")

        loop.set_postfix(D=f"{d_loss:.3f}", G=f"{g_loss:.3f}")


def main():
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)

    disc_A = Discriminator().to(config.DEVICE)
    disc_B = Discriminator().to(config.DEVICE)
    gen_A  = Generator().to(config.DEVICE)
    gen_B  = Generator().to(config.DEVICE)

    opt_disc = optim.Adam(
        list(disc_A.parameters()) + list(disc_B.parameters()),
        lr=config.LEARNING_RATE, betas=(0.5, 0.999)
    )
    opt_gen = optim.Adam(
        list(gen_A.parameters()) + list(gen_B.parameters()),
        lr=config.LEARNING_RATE, betas=(0.5, 0.999)
    )

    l1  = nn.L1Loss()
    mse = nn.MSELoss()

    if config.LOAD_MODEL:
        load_checkpoint(f"{config.CHECKPOINT_DIR}/gen_A.pth", gen_A, opt_gen, config.LEARNING_RATE)
        load_checkpoint(f"{config.CHECKPOINT_DIR}/gen_B.pth", gen_B, opt_gen, config.LEARNING_RATE)
        load_checkpoint(f"{config.CHECKPOINT_DIR}/disc_A.pth", disc_A, opt_disc, config.LEARNING_RATE)
        load_checkpoint(f"{config.CHECKPOINT_DIR}/disc_B.pth", disc_B, opt_disc, config.LEARNING_RATE)

    dataset = ArtDataset(config.TRAIN_DIR_A, config.TRAIN_DIR_B, transform=transforms)
    loader  = DataLoader(dataset, batch_size=config.BATCH_SIZE,
                         shuffle=True, num_workers=config.NUM_WORKERS, pin_memory=True)

    d_scaler = torch.amp.GradScaler('cuda')
    g_scaler = torch.amp.GradScaler('cuda')

    for epoch in range(1, config.NUM_EPOCHS + 1):
        train_epoch(disc_A, disc_B, gen_A, gen_B, loader,
                    opt_disc, opt_gen, l1, mse, d_scaler, g_scaler, epoch)

        if config.SAVE_MODEL:
            save_checkpoint(gen_A,  opt_gen,  f"{config.CHECKPOINT_DIR}/gen_A.pth")
            save_checkpoint(gen_B,  opt_gen,  f"{config.CHECKPOINT_DIR}/gen_B.pth")
            save_checkpoint(disc_A, opt_disc, f"{config.CHECKPOINT_DIR}/disc_A.pth")
            save_checkpoint(disc_B, opt_disc, f"{config.CHECKPOINT_DIR}/disc_B.pth")
        
        print(f"Epoch {epoch}/{config.NUM_EPOCHS} terminée — checkpoints sauvegardés")


if __name__ == "__main__":
    main()