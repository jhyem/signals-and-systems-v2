import argparse
import csv
import json
import os
import random
import time

import numpy as np
import soundfile as sf
import torch
from torch.utils.data import Dataset, DataLoader

from src.model import MaskCNN

try:
    from pesq import pesq
    PESQ_AVAILABLE = True
except ImportError:
    pesq = None
    PESQ_AVAILABLE = False


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
    waveform, sample_rate = sf.read(path, dtype="float32")

    if waveform.ndim == 2:
        waveform = waveform.mean(axis=1)

    return waveform, sample_rate


# =========================================================
# 3. Paired dataset
# =========================================================

class PairedAudioDataset(Dataset):

    def __init__(self, clean_dir, noisy_dir, sample_rate=16000):
        self.clean_dir = clean_dir
        self.noisy_dir = noisy_dir
        self.sample_rate = sample_rate

        noisy_files = sorted(
            f for f in os.listdir(noisy_dir)
            if f.lower().endswith(".wav")
        )

        self.pairs = []

        for noisy_file in noisy_files:
            clean_file = noisy_file.replace("_noisy.wav", ".wav")

            clean_path = os.path.join(clean_dir, clean_file)
            noisy_path = os.path.join(noisy_dir, noisy_file)

            if os.path.isfile(clean_path):
                self.pairs.append((clean_path, noisy_path))

        print(f"Dataset pairs: {len(self.pairs)}")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        clean_path, noisy_path = self.pairs[index]

        clean, sr_clean = read_audio(clean_path)
        noisy, sr_noisy = read_audio(noisy_path)

        if sr_clean != self.sample_rate or sr_noisy != self.sample_rate:
            raise ValueError(
                f"Sample rate mismatch: {clean_path}"
            )

        if clean.shape != noisy.shape:
            raise ValueError(
                f"Audio length mismatch: {clean_path}"
            )

        return (
            torch.from_numpy(noisy.copy()),
            torch.from_numpy(clean.copy())
        )


# =========================================================
# 4. Spectral loss
# =========================================================

def spectral_loss(enhanced, clean, n_fft=512, hop=128):

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

    enhanced_mag = torch.log1p(enhanced_stft.abs())
    clean_mag = torch.log1p(clean_stft.abs())

    return torch.mean(torch.abs(enhanced_mag - clean_mag))


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
    pesq_failures = 0

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

            if PESQ_AVAILABLE:

                enhanced_cpu = enhanced.detach().cpu().numpy()
                clean_cpu = clean.detach().cpu().numpy()

                for i in range(batch_size):

                    try:
                        score = pesq(
                            sample_rate,
                            clean_cpu[i],
                            enhanced_cpu[i],
                            "wb"
                        )

                        if np.isfinite(score):
                            pesq_scores.append(float(score))

                    except Exception as e:
                        pesq_failures += 1

                        if pesq_failures <= 3:
                            print(
                                "PESQ sample failure:",
                                type(e).__name__,
                                str(e)
                            )

    avg_loss = total_loss / max(total_samples, 1)

    avg_pesq = (
        float(np.mean(pesq_scores))
        if pesq_scores
        else None
    )

    print(
        f"Validation PESQ samples: "
        f"{len(pesq_scores)} successful, "
        f"{pesq_failures} failed"
    )

    return avg_loss, avg_pesq


# =========================================================
# 6. Main
# =========================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--data-dir", required=True)
    parser.add_argument(
        "--config",
        default="configs/baseline.json"
    )
    parser.add_argument(
        "--output-dir",
        default="results/baseline"
    )
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = json.load(f)

    training_config = config.get("training", {})
    model_config = config.get("model", {})

    seed = config.get("seed", args.seed)

    sample_rate = config.get("sample_rate", 16000)
    n_fft = config.get("n_fft", 512)
    hop = config.get("hop_length", 128)

    batch_size = training_config.get("batch_size", 16)
    initial_lr = training_config.get("initial_lr", 0.001)
    max_epochs = training_config.get("max_epochs", 200)
    min_epochs = training_config.get("min_epochs", 30)

    patience = training_config.get(
        "early_stopping_patience", 20
    )

    min_delta = training_config.get("min_delta", 0.001)

    scheduler_factor = training_config.get(
        "lr_scheduler_factor", 0.5
    )

    scheduler_patience = training_config.get(
        "lr_scheduler_patience", 5
    )

    min_lr = training_config.get("min_lr", 1e-5)

    hidden_channels = model_config.get(
        "hidden_channels", [64, 64]
    )

    set_seed(seed)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("=" * 60)
    print("Baseline MaskCNN Training")
    print("=" * 60)

    print("Experiment:", config.get("experiment_name"))
    print("Device:", device)

    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

    print("Seed:", seed)
    print("Sample rate:", sample_rate)
    print("FFT:", n_fft)
    print("Hop:", hop)
    print("Batch size:", batch_size)
    print("Initial LR:", initial_lr)
    print("Max epochs:", max_epochs)
    print("Min epochs:", min_epochs)
    print("Early stopping patience:", patience)
    print("Min delta:", min_delta)
    print("Scheduler factor:", scheduler_factor)
    print("Scheduler patience:", scheduler_patience)
    print("Min LR:", min_lr)
    print("Hidden channels:", hidden_channels)
    print("PESQ available:", PESQ_AVAILABLE)

    train_dataset = PairedAudioDataset(
        os.path.join(args.data_dir, "train", "clean"),
        os.path.join(args.data_dir, "train", "noisy"),
        sample_rate
    )

    val_dataset = PairedAudioDataset(
        os.path.join(args.data_dir, "validation", "clean"),
        os.path.join(args.data_dir, "validation", "noisy"),
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

    model = MaskCNN(
        n_fft=n_fft,
        hop_length=hop,
        hidden_channels=hidden_channels
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=initial_lr
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=scheduler_factor,
        patience=scheduler_patience,
        threshold=min_delta,
        min_lr=min_lr
    )

    os.makedirs(args.output_dir, exist_ok=True)

    checkpoint_path = os.path.join(
        args.output_dir,
        "best_model.pt"
    )

    log_path = os.path.join(
        args.output_dir,
        "training_log.csv"
    )

    # 이전 실패 학습의 체크포인트 제거
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)

    history = []

    best_metric = None
    best_epoch = 0
    best_metric_mode = None

    no_improvement = 0

    start_time = time.time()

    for epoch in range(1, max_epochs + 1):

        model.train()

        total_train_loss = 0.0
        total_train_samples = 0

        for noisy, clean in train_loader:

            noisy = noisy.to(device)
            clean = clean.to(device)

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

            total_train_loss += (
                loss.item() * batch_size_actual
            )

            total_train_samples += batch_size_actual

        train_loss = (
            total_train_loss /
            max(total_train_samples, 1)
        )

        val_loss, val_pesq = evaluate(
            model,
            val_loader,
            device,
            n_fft,
            hop,
            sample_rate
        )

        # PESQ를 사용할 수 있으면 PESQ 최대화,
        # 그렇지 않으면 Validation Loss 최소화
        metric_mode = (
            "PESQ" if val_pesq is not None else "LOSS"
        )

        metric = (
            val_pesq if metric_mode == "PESQ"
            else val_loss
        )

        # 평가 기준이 바뀌면 비교 기준 초기화
        if best_metric_mode != metric_mode:

            best_metric = None
            best_epoch = 0
            no_improvement = 0
            best_metric_mode = metric_mode

            print(
                f"Metric mode changed to: {metric_mode}"
            )

        if metric_mode == "PESQ":
            improved = (
                best_metric is None
                or metric > best_metric + min_delta
            )
        else:
            improved = (
                best_metric is None
                or metric < best_metric - min_delta
            )

        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch {epoch:03d}/{max_epochs} | "
            f"Train Loss: {train_loss:.6f} | "
            f"Val Loss: {val_loss:.6f} | "
            f"Val PESQ: "
            f"{val_pesq if val_pesq is not None else 'N/A'} | "
            f"LR: {current_lr:.6f}"
        )

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_pesq": val_pesq,
            "learning_rate": current_lr
        })

        if improved:

            best_metric = metric
            best_epoch = epoch
            no_improvement = 0

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "config": config,
                    "seed": seed,
                    "val_loss": val_loss,
                    "val_pesq": val_pesq,
                    "best_metric": best_metric,
                    "metric_mode": metric_mode
                },
                checkpoint_path
            )

            print(
                f"  -> Best model saved "
                f"({metric_mode}: {best_metric:.6f})"
            )

        else:
            no_improvement += 1

        # Scheduler uses the same metric direction
        scheduler.step(metric)

        # CSV is updated every epoch
        with open(
            log_path,
            "w",
            newline="",
            encoding="utf-8"
        ) as f:

            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "epoch",
                    "train_loss",
                    "val_loss",
                    "val_pesq",
                    "learning_rate"
                ]
            )

            writer.writeheader()
            writer.writerows(history)

        if (
            epoch >= min_epochs
            and no_improvement >= patience
        ):

            print(
                f"Early stopping at epoch {epoch}"
            )
            break

    elapsed = time.time() - start_time

    print("\n" + "=" * 60)
    print("Training finished")
    print("=" * 60)

    print("Best epoch:", best_epoch)
    print("Best metric:", best_metric)
    print("Metric mode:", best_metric_mode)
    print(f"Elapsed time: {elapsed:.2f} seconds")
    print("Checkpoint:", checkpoint_path)
    print("Training log:", log_path)

    if not os.path.isfile(checkpoint_path):
        raise RuntimeError(
            "Best checkpoint was not created."
        )


if __name__ == "__main__":
    main()
