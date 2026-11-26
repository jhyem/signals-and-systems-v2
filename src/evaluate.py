import argparse
import csv
import json
import os

import numpy as np
import soundfile as sf
import torch

from src.model import MaskCNN


def read_audio(path):
    waveform, sr = sf.read(path, dtype="float32")

    if waveform.ndim == 2:
        waveform = waveform.mean(axis=1)

    return waveform, sr


def calculate_snr(clean, estimate):
    clean = np.asarray(clean, dtype=np.float64)
    estimate = np.asarray(estimate, dtype=np.float64)

    noise = clean - estimate

    signal_power = np.sum(clean ** 2)
    noise_power = np.sum(noise ** 2)

    if noise_power <= 1e-12:
        return float("inf")

    if signal_power <= 1e-12:
        return float("-inf")

    return float(
        10.0 * np.log10(signal_power / noise_power)
    )


def calculate_pesq(clean, estimate, sample_rate=16000):
    from pesq import pesq

    return float(
        pesq(
            sample_rate,
            clean,
            estimate,
            "wb",
        )
    )


def find_pairs(clean_dir, noisy_dir):
    clean_files = {
        f for f in os.listdir(clean_dir)
        if f.lower().endswith(".wav")
    }

    noisy_files = {
        f for f in os.listdir(noisy_dir)
        if f.lower().endswith(".wav")
    }

    pairs = []

    for noisy_name in sorted(noisy_files):
        stem = noisy_name[:-4]

        if stem.endswith("_noisy"):
            stem = stem[:-6]

        clean_name = stem + ".wav"

        if clean_name in clean_files:
            pairs.append(
                (clean_name, noisy_name)
            )

    if not pairs:
        raise RuntimeError("No paired WAV files found.")

    return pairs


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="validation")
    parser.add_argument("--output-dir", default="results/evaluation")

    args = parser.parse_args()

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    checkpoint = torch.load(
        args.checkpoint,
        map_location=device,
        weights_only=False,
    )

    config = checkpoint.get("config", {})

    n_fft = config.get("n_fft", 512)
    hop = config.get("hop_length", 128)
    hidden_channels = tuple(
        config.get("hidden_channels", [64, 64])
    )
    sample_rate = config.get("sample_rate", 16000)

    model = MaskCNN(
        n_fft=n_fft,
        hop_length=hop,
        hidden_channels=hidden_channels,
    ).to(device)

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    clean_dir = os.path.join(
        args.data_dir, args.split, "clean"
    )

    noisy_dir = os.path.join(
        args.data_dir, args.split, "noisy"
    )

    pairs = find_pairs(clean_dir, noisy_dir)

    os.makedirs(args.output_dir, exist_ok=True)

    csv_path = os.path.join(
        args.output_dir, "evaluation.csv"
    )

    summary_path = os.path.join(
        args.output_dir, "summary.json"
    )

    results = []

    print("Device:", device)
    print("Split:", args.split)
    print("Pairs:", len(pairs))

    for index, (clean_name, noisy_name) in enumerate(pairs, 1):
        clean, clean_sr = read_audio(
            os.path.join(clean_dir, clean_name)
        )

        noisy, noisy_sr = read_audio(
            os.path.join(noisy_dir, noisy_name)
        )

        if clean_sr != sample_rate or noisy_sr != sample_rate:
            raise ValueError(
                f"Sample rate mismatch: {clean_name}"
            )

        if len(clean) != len(noisy):
            raise ValueError(
                f"Length mismatch: {clean_name}"
            )

        noisy_snr = calculate_snr(clean, noisy)

        try:
            noisy_pesq = calculate_pesq(
                clean, noisy, sample_rate
            )
        except Exception as e:
            print(
                f"Noisy PESQ failed: {noisy_name}: {e}"
            )
            noisy_pesq = None

        noisy_tensor = torch.from_numpy(
            noisy
        ).unsqueeze(0).to(device)

        with torch.no_grad():
            enhanced_tensor = model(noisy_tensor)

        enhanced = (
            enhanced_tensor.squeeze(0)
            .cpu()
            .numpy()
        )

        if len(enhanced) != len(clean):
            raise RuntimeError(
                f"Output length mismatch: {clean_name}"
            )

        enhanced_snr = calculate_snr(
            clean, enhanced
        )

        try:
            enhanced_pesq = calculate_pesq(
                clean, enhanced, sample_rate
            )
        except Exception as e:
            print(
                f"Enhanced PESQ failed: {noisy_name}: {e}"
            )
            enhanced_pesq = None

        results.append({
            "filename": clean_name,
            "noisy_snr": noisy_snr,
            "enhanced_snr": enhanced_snr,
            "noisy_pesq": noisy_pesq,
            "enhanced_pesq": enhanced_pesq,
        })

        print(
            f"[{index}/{len(pairs)}] {clean_name} | "
            f"SNR: {noisy_snr:.3f} -> {enhanced_snr:.3f} | "
            f"PESQ: {noisy_pesq} -> {enhanced_pesq}"
        )

    fields = [
        "filename",
        "noisy_snr",
        "enhanced_snr",
        "noisy_pesq",
        "enhanced_pesq",
    ]

    with open(
        csv_path, "w", newline="", encoding="utf-8"
    ) as f:
        writer = csv.DictWriter(
            f, fieldnames=fields
        )
        writer.writeheader()
        writer.writerows(results)

    def average(key):
        values = [
            row[key]
            for row in results
            if row[key] is not None
            and np.isfinite(row[key])
        ]

        return (
            float(np.mean(values))
            if values else None
        )

    summary = {
        "checkpoint": args.checkpoint,
        "split": args.split,
        "sample_count": len(results),
        "mean_noisy_snr": average("noisy_snr"),
        "mean_enhanced_snr": average("enhanced_snr"),
        "mean_noisy_pesq": average("noisy_pesq"),
        "mean_enhanced_pesq": average("enhanced_pesq"),
    }

    summary["snr_improvement"] = (
        summary["mean_enhanced_snr"]
        - summary["mean_noisy_snr"]
        if summary["mean_enhanced_snr"] is not None
        and summary["mean_noisy_snr"] is not None
        else None
    )

    summary["pesq_improvement"] = (
        summary["mean_enhanced_pesq"]
        - summary["mean_noisy_pesq"]
        if summary["mean_enhanced_pesq"] is not None
        and summary["mean_noisy_pesq"] is not None
        else None
    )

    with open(
        summary_path, "w", encoding="utf-8"
    ) as f:
        json.dump(
            summary,
            f,
            ensure_ascii=False,
            indent=4,
        )

    print("\n===== Evaluation Summary =====")

    for key, value in summary.items():
        print(f"{key}: {value}")

    print("\nSaved:")
    print(csv_path)
    print(summary_path)


if __name__ == "__main__":
    main()
