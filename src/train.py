import argparse
import csv
import json
import os
import random
import time

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from src.model import MaskCNN


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def read_audio(path):
    waveform, sr = sf.read(path, dtype="float32")

    if waveform.ndim == 2:
        waveform = waveform.mean(axis=1)

    return waveform, sr


class PairedAudioDataset(Dataset):
    def __init__(self, clean_dir, noisy_dir, sample_rate=16000):
        self.clean_dir = clean_dir
        self.noisy_dir = noisy_dir
        self.sample_rate = sample_rate
        self.pairs = []

        clean_files = {
            f for f in os.listdir(clean_dir)
            if f.lower().endswith(".wav")
        }

        noisy_files = {
            f for f in os.listdir(noisy_dir)
            if f.lower().endswith(".wav")
        }

        for noisy_name in sorted(noisy_files):
            stem = noisy_name[:-4]

            if stem.endswith("_noisy"):
                stem = stem[:-6]

            clean_name = stem + ".wav"

            if clean_name in clean_files:
                self.pairs.append(
                    (clean_name, noisy_name)
                )

        if not self.pairs:
            raise RuntimeError(
                f"No paired WAV files found: {clean_dir}, {noisy_dir}"
            )

        print(f"Dataset pairs: {len(self.pairs)}")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        clean_name, noisy_name = self.pairs[idx]

        clean, clean_sr = read_audio(
            os.path.join(self.clean_dir, clean_name)
        )

        noisy, noisy_sr = read_audio(
            os.path.join(self.noisy_dir, noisy_name)
        )

        if clean_sr != self.sample_rate or noisy_sr != self.sample_rate:
            raise ValueError(
                f"Sample rate mismatch: {clean_sr}, {noisy_sr}"
            )

        if len(clean) != len(noisy):
            raise ValueError(
                f"Length mismatch: {clean_name}, {noisy_name}"
            )

        return (
            torch.from_numpy(clean),
            torch.from_numpy(noisy),
        )


def spectral_loss(enhanced, clean, n_fft=512, hop=128):
    window = torch.hann_window(
        n_fft,
        device=enhanced.device,
    )

    enhanced_stft = torch.stft(
        enhanced,
        n_fft=n_fft,
        hop_length=hop,
        window=window,
        return_complex=True,
    )

    clean_stft = torch.stft(
        clean,
        n_fft=n_fft,
        hop_length=hop,
        window=window,
        return_complex=True,
    )

    return F.l1_loss(
        torch.log1p(enhanced_stft.abs()),
        torch.log1p(clean_stft.abs()),
    )


def evaluate(model, loader, device, n_fft, hop):
    model.eval()

    total_loss = 0.0
    count = 0
    pesq_scores = []

    try:
        from pesq import pesq
        pesq_available = True
    except ImportError:
        pesq_available = False

    with torch.no_grad():
        for clean, noisy in loader:
            clean = clean.to(device)
            noisy = noisy.to(device)

            enhanced = model(noisy)

            loss = spectral_loss(
                enhanced, clean, n_fft, hop
            )

            total_loss += loss.item() * clean.size(0)
            count += clean.size(0)

            if pesq_available:
                for i in range(clean.size(0)):
                    try:
                        score = pesq(
                            16000,
                            clean[i].cpu().numpy(),
                            enhanced[i].cpu().numpy(),
                            "wb",
                        )
                        pesq_scores.append(score)
                    except Exception:
                        pass

    avg_loss = total_loss / count

    avg_pesq = (
        float(np.mean(pesq_scores))
        if pesq_scores else None
    )

    return avg_loss, avg_pesq


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--config", default="configs/baseline.json")
    parser.add_argument("--output-dir", default="results/baseline")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = json.load(f)

    seed = args.seed
    set_seed(seed)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    sample_rate = config.get("sample_rate", 16000)
    n_fft = config.get("n_fft", 512)
    hop = config.get("hop_length", 128)

    batch_size = config.get("batch_size", 16)
    lr = config.get("learning_rate", 0.001)
    max_epochs = config.get("max_epochs", 200)
    min_epochs = config.get("min_epochs", 30)
    patience = config.get("patience", 20)
    min_delta = config.get("min_delta", 0.001)

    hidden_channels = tuple(
        config.get("hidden_channels", [64, 64])
    )

    train_dataset = PairedAudioDataset(
        os.path.join(args.data_dir, "train", "clean"),
        os.path.join(args.data_dir, "train", "noisy"),
        sample_rate,
    )

    val_dataset = PairedAudioDataset(
        os.path.join(args.data_dir, "validation", "clean"),
        os.path.join(args.data_dir, "validation", "noisy"),
        sample_rate,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    model = MaskCNN(
        n_fft=n_fft,
        hop_length=hop,
        hidden_channels=hidden_channels,
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=config.get("lr_patience", 5),
        threshold=min_delta,
        min_lr=config.get("min_lr", 1e-5),
    )

    os.makedirs(args.output_dir, exist_ok=True)

    best_path = os.path.join(
        args.output_dir, "best_model.pt"
    )

    log_path = os.path.join(
        args.output_dir, "training_log.csv"
    )

    best_metric = -float("inf")
    best_val_loss = float("inf")
    early_stop_counter = 0

    start_time = time.time()

    with open(log_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "epoch",
            "train_loss",
            "val_loss",
            "val_pesq",
            "learning_rate",
        ])

    print("Device:", device)
    print("Train samples:", len(train_dataset))
    print("Validation samples:", len(val_dataset))

    for epoch in range(1, max_epochs + 1):
        model.train()

        train_loss_sum = 0.0
        train_count = 0

        for clean, noisy in train_loader:
            clean = clean.to(device)
            noisy = noisy.to(device)

            optimizer.zero_grad()

            enhanced = model(noisy)

            loss = spectral_loss(
                enhanced, clean, n_fft, hop
            )

            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item() * clean.size(0)
            train_count += clean.size(0)

        train_loss = train_loss_sum / train_count

        val_loss, val_pesq = evaluate(
            model, val_loader, device, n_fft, hop
        )

        if val_pesq is not None:
            metric = val_pesq
            scheduler.step(metric)
        else:
            metric = -val_loss
            scheduler.step(metric)

        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch {epoch:03d}/{max_epochs} | "
            f"Train Loss: {train_loss:.5f} | "
            f"Val Loss: {val_loss:.5f} | "
            f"PESQ: {val_pesq if val_pesq is not None else 'N/A'} | "
            f"LR: {current_lr:.6f}"
        )

        with open(log_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch,
                train_loss,
                val_loss,
                val_pesq,
                current_lr,
            ])

        improved = metric > best_metric + min_delta

        if improved:
            best_metric = metric
            best_val_loss = val_loss
            early_stop_counter = 0

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "config": config,
                    "seed": seed,
                    "val_loss": val_loss,
                    "val_pesq": val_pesq,
                },
                best_path,
            )

            print("  -> Best model saved")
        else:
            early_stop_counter += 1

        if (
            epoch >= min_epochs
            and early_stop_counter >= patience
        ):
            print("Early stopping.")
            break

    elapsed = time.time() - start_time

    print("\nTraining finished.")
    print("Best checkpoint:", best_path)
    print("Best metric:", best_metric)
    print("Best validation loss:", best_val_loss)
    print(f"Elapsed time: {elapsed:.1f} sec")


if __name__ == "__main__":
    main()
