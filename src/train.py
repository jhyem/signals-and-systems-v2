

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


# =========================================================
# 1. Seed
# =========================================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# =========================================================
# 2. Audio loading
# =========================================================

def read_audio(path):
    waveform, sample_rate = sf.read(
        path,
        dtype="float32"
    )

    # Stereo -> mono
    if waveform.ndim == 2:
        waveform = waveform.mean(axis=1)

    return waveform, sample_rate


# =========================================================
# 3. Paired dataset
# =========================================================

class PairedAudioDataset(Dataset):

    def __init__(
        self,
        clean_dir,
        noisy_dir,
        sample_rate=16000
    ):
        self.clean_dir = clean_dir
        self.noisy_dir = noisy_dir
        self.sample_rate = sample_rate

        clean_files = sorted(
            f for f in os.listdir(clean_dir)
            if f.lower().endswith(".wav")
        )

        noisy_files = sorted(
            f for f in os.listdir(noisy_dir)
            if f.lower().endswith(".wav")
        )

        noisy_map = {}

        for noisy_filename in noisy_files:
            stem = os.path.splitext(noisy_filename)[0]

            if stem.endswith("_noisy"):
                stem = stem[:-6]

            noisy_map[stem] = noisy_filename

        self.pairs = []

        for clean_filename in clean_files:
            stem = os.path.splitext(clean_filename)[0]

            if stem in noisy_map:
                self.pairs.append(
                    (
                        os.path.join(clean_dir, clean_filename),
                        os.path.join(
                            noisy_dir,
                            noisy_map[stem]
                        )
                    )
                )

        print(f"Dataset pairs: {len(self.pairs)}")

        if len(self.pairs) == 0:
            raise RuntimeError(
                "No clean/noisy audio pairs found."
            )

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):

        clean_path, noisy_path = self.pairs[index]

        clean, clean_sr = read_audio(clean_path)
        noisy, noisy_sr = read_audio(noisy_path)

        if clean_sr != self.sample_rate:
            raise ValueError(
                f"Clean sample rate mismatch: {clean_path}"
            )

        if noisy_sr != self.sample_rate:
            raise ValueError(
                f"Noisy sample rate mismatch: {noisy_path}"
            )

        if len(clean) != len(noisy):
            raise ValueError(
                f"Length mismatch: {clean_path}"
            )

        clean = torch.from_numpy(clean.copy())
        noisy = torch.from_numpy(noisy.copy())

        return noisy, clean


# =========================================================
# 4. Loss
# =========================================================

def spectral_loss(
    enhanced,
    clean,
    n_fft=512,
    hop=128
):

    window = torch.hann_window(
        n_fft,
        device=enhanced.device
    )

    enhanced_stft = torch.stft(
        enhanced,
        n_fft=n_fft,
        hop_length=hop,
        window=window,
        return_complex=True
    )

    clean_stft = torch.stft(
        clean,
        n_fft=n_fft,
        hop_length=hop,
        window=window,
        return_complex=True
    )

    enhanced_mag = enhanced_stft.abs()
    clean_mag = clean_stft.abs()

    loss = F.l1_loss(
        torch.log1p(enhanced_mag),
        torch.log1p(clean_mag)
    )

    return loss


# =========================================================
# 5. Validation
# =========================================================

def evaluate(
    model,
    loader,
    device,
    n_fft,
    hop,
    sample_rate=16000
):

    model.eval()

    total_loss = 0.0
    total_samples = 0

    pesq_scores = []

    try:
        from pesq import pesq as pesq_fn
        pesq_available = True
    except ImportError:
        pesq_available = False
        pesq_fn = None

    with torch.no_grad():

        for noisy, clean in loader:

            noisy = noisy.to(device)
            clean = clean.to(device)

            enhanced = model(noisy)

            loss = spectral_loss(
                enhanced,
                clean,
                n_fft,
                hop
            )

            batch_size = noisy.size(0)

            total_loss += loss.item() * batch_size
            total_samples += batch_size

            if pesq_available:

                enhanced_np = (
                    enhanced.detach()
                    .cpu()
                    .numpy()
                )

                clean_np = (
                    clean.detach()
                    .cpu()
                    .numpy()
                )

                for i in range(batch_size):

                    try:
                        score = pesq_fn(
                            sample_rate,
                            clean_np[i],
                            enhanced_np[i],
                            "wb"
                        )

                        pesq_scores.append(score)

                    except Exception as e:
                        print(
                            f"PESQ error: {e}"
                        )

    avg_loss = (
        total_loss / total_samples
        if total_samples > 0
        else float("inf")
    )

    avg_pesq = (
        float(np.mean(pesq_scores))
        if len(pesq_scores) > 0
        else None
    )

    return avg_loss, avg_pesq


# =========================================================
# 6. Main training
# =========================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data-dir",
        required=True
    )

    parser.add_argument(
        "--config",
        default="configs/baseline.json"
    )

    parser.add_argument(
        "--output-dir",
        default="results/baseline"
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42
    )

    args = parser.parse_args()

    # -----------------------------------------------------
    # Config loading
    # -----------------------------------------------------

    with open(
        args.config,
        "r",
        encoding="utf-8"
    ) as f:

        config = json.load(f)

    training_config = config.get(
        "training",
        {}
    )

    model_config = config.get(
        "model",
        {}
    )

    seed = config.get(
        "seed",
        args.seed
    )

    set_seed(seed)

    # -----------------------------------------------------
    # Settings
    # -----------------------------------------------------

    sample_rate = config.get(
        "sample_rate",
        16000
    )

    n_fft = config.get(
        "n_fft",
        512
    )

    hop = config.get(
        "hop_length",
        128
    )

    batch_size = training_config.get(
        "batch_size",
        16
    )

    lr = training_config.get(
        "initial_lr",
        0.001
    )

    max_epochs = training_config.get(
        "max_epochs",
        200
    )

    min_epochs = training_config.get(
        "min_epochs",
        30
    )

    patience = training_config.get(
        "early_stopping_patience",
        20
    )

    min_delta = training_config.get(
        "min_delta",
        0.001
    )

    lr_factor = training_config.get(
        "lr_scheduler_factor",
        0.5
    )

    lr_patience = training_config.get(
        "lr_scheduler_patience",
        5
    )

    min_lr = training_config.get(
        "min_lr",
        1e-5
    )

    hidden_channels = model_config.get(
        "hidden_channels",
        [64, 64]
    )

    # -----------------------------------------------------
    # Device
    # -----------------------------------------------------

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("=" * 60)
    print("Baseline MaskCNN Training")
    print("=" * 60)

    print("Experiment:", config.get("experiment_name"))
    print("Device:", device)

    if torch.cuda.is_available():
        print(
            "GPU:",
            torch.cuda.get_device_name(0)
        )

    print("Seed:", seed)
    print("Sample rate:", sample_rate)
    print("FFT:", n_fft)
    print("Hop:", hop)
    print("Batch size:", batch_size)
    print("Initial LR:", lr)
    print("Max epochs:", max_epochs)
    print("Min epochs:", min_epochs)
    print("Early stopping patience:", patience)
    print("Min delta:", min_delta)
    print("Scheduler factor:", lr_factor)
    print("Scheduler patience:", lr_patience)
    print("Min LR:", min_lr)
    print("Hidden channels:", hidden_channels)

    # -----------------------------------------------------
    # Dataset
    # -----------------------------------------------------

    train_clean_dir = os.path.join(
        args.data_dir,
        "train",
        "clean"
    )

    train_noisy_dir = os.path.join(
        args.data_dir,
        "train",
        "noisy"
    )

    val_clean_dir = os.path.join(
        args.data_dir,
        "validation",
        "clean"
    )

    val_noisy_dir = os.path.join(
        args.data_dir,
        "validation",
        "noisy"
    )

    train_dataset = PairedAudioDataset(
        train_clean_dir,
        train_noisy_dir,
        sample_rate
    )

    val_dataset = PairedAudioDataset(
        val_clean_dir,
        val_noisy_dir,
        sample_rate
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available()
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available()
    )

    # -----------------------------------------------------
    # Model
    # -----------------------------------------------------

    model = MaskCNN(
        n_fft=n_fft,
        hop_length=hop,
        hidden_channels=tuple(hidden_channels)
    ).to(device)

    # -----------------------------------------------------
    # Optimizer
    # -----------------------------------------------------

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr
    )

    # -----------------------------------------------------
    # Scheduler
    # -----------------------------------------------------

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=lr_factor,
        patience=lr_patience,
        threshold=min_delta,
        min_lr=min_lr
    )

    # -----------------------------------------------------
    # Output
    # -----------------------------------------------------

    os.makedirs(
        args.output_dir,
        exist_ok=True
    )

    checkpoint_path = os.path.join(
        args.output_dir,
        "best_model.pt"
    )

    log_path = os.path.join(
        args.output_dir,
        "training_log.csv"
    )

    # -----------------------------------------------------
    # PESQ availability
    # -----------------------------------------------------

    try:
        from pesq import pesq
        metric_mode = "PESQ"
    except ImportError:
        metric_mode = "LOSS"

    if metric_mode == "PESQ":
        best_metric = -float("inf")
    else:
        best_metric = float("inf")

    best_epoch = 0
    early_stop_counter = 0

    # -----------------------------------------------------
    # CSV log
    # -----------------------------------------------------

    with open(
        log_path,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.writer(f)

        writer.writerow([
            "epoch",
            "train_loss",
            "val_loss",
            "val_pesq",
            "learning_rate"
        ])

    # -----------------------------------------------------
    # Training loop
    # -----------------------------------------------------

    start_time = time.time()

    for epoch in range(
        1,
        max_epochs + 1
    ):

        model.train()

        train_loss_sum = 0.0
        train_samples = 0

        for noisy, clean in train_loader:

            noisy = noisy.to(
                device,
                non_blocking=True
            )

            clean = clean.to(
                device,
                non_blocking=True
            )

            optimizer.zero_grad()

            enhanced = model(noisy)

            loss = spectral_loss(
                enhanced,
                clean,
                n_fft,
                hop
            )

            loss.backward()

            optimizer.step()

            batch_size_actual = noisy.size(0)

            train_loss_sum += (
                loss.item() * batch_size_actual
            )

            train_samples += batch_size_actual

        train_loss = (
            train_loss_sum / train_samples
        )

        # -------------------------------------------------
        # Validation
        # -------------------------------------------------

        val_loss, val_pesq = evaluate(
            model,
            val_loader,
            device,
            n_fft,
            hop,
            sample_rate
        )

        # -------------------------------------------------
        # Metric
        # -------------------------------------------------

        if val_pesq is not None:

            metric = val_pesq
            metric_mode = "PESQ"

        else:

            metric = -val_loss
            metric_mode = "LOSS"

        # -------------------------------------------------
        # Scheduler
        # -------------------------------------------------

        scheduler.step(metric)

        current_lr = optimizer.param_groups[0]["lr"]

        # -------------------------------------------------
        # Logging
        # -------------------------------------------------

        print(
            f"Epoch {epoch:03d}/{max_epochs} | "
            f"Train Loss: {train_loss:.6f} | "
            f"Val Loss: {val_loss:.6f} | "
            f"Val PESQ: "
            f"{val_pesq if val_pesq is not None else 'N/A'} | "
            f"LR: {current_lr:.6f}"
        )

        with open(
            log_path,
            "a",
            newline="",
            encoding="utf-8"
        ) as f:

            writer = csv.writer(f)

            writer.writerow([
                epoch,
                train_loss,
                val_loss,
                val_pesq,
                current_lr
            ])

        # -------------------------------------------------
        # Best model
        # -------------------------------------------------

        if metric_mode == "PESQ":

            improved = (
                metric > best_metric + min_delta
            )

        else:

            improved = (
                metric > best_metric + min_delta
            )

        if improved:

            best_metric = metric
            best_epoch = epoch
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
                    "best_metric": best_metric
                },
                checkpoint_path
            )

            print(
                f"  -> Best model saved "
                f"(epoch {epoch})"
            )

        else:

            early_stop_counter += 1

        # -------------------------------------------------
        # Early stopping
        # -------------------------------------------------

        if (
            epoch >= min_epochs
            and early_stop_counter >= patience
        ):

            print(
                f"\nEarly stopping at epoch {epoch}"
            )

            break

    # -----------------------------------------------------
    # Finished
    # -----------------------------------------------------

    elapsed = time.time() - start_time

    print("\n" + "=" * 60)
    print("Training finished")
    print("=" * 60)

    print("Best epoch:", best_epoch)
    print("Best metric:", best_metric)
    print("Metric mode:", metric_mode)
    print("Elapsed time:", round(elapsed, 2), "seconds")
    print("Checkpoint:", checkpoint_path)
    print("Training log:", log_path)


if __name__ == "__main__":
    main()
