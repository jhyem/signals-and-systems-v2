# Shared Deep Learning Baseline Results

## 1. 목적

본 문서는 딥러닝 기반 음성 잡음 제거 챌린지의 팀 공통 Baseline 모델과 Validation 평가 결과를 기록한다.

팀원들은 본 Baseline을 기준으로 모델 구조, 학습 방법, 손실 함수, 데이터 전처리 및 기타 개선 기법을 독립적으로 실험한다.

## 2. Dataset

| 항목 | 내용 |
|---|---|
| Sample rate | 16,000 Hz |
| Audio length | 3 seconds |
| Samples per audio | 48,000 |
| Channel | Mono |
| Train pairs | 1,296 |
| Validation pairs | 164 |

## 3. Baseline Model

- Model: MaskCNN
- Input: Noisy audio waveform
- STFT: n_fft=512, hop_length=128, Hann window
- Input feature: Log magnitude spectrum (log1p)
- Output: Time-frequency magnitude mask
- Mask range: 0.0 ~ 1.0
- Reconstruction: Apply predicted mask to noisy complex STFT and perform inverse STFT
- Phase: Noisy speech phase is retained

### Model Architecture

| Layer | Configuration |
|---|---|
| Conv1d | 257 → 64, kernel_size=5, padding=2 |
| ReLU | Activation |
| Conv1d | 64 → 64, kernel_size=3, padding=1 |
| ReLU | Activation |
| Conv1d | 64 → 257, kernel_size=1 |
| Sigmoid | Mask output |

## 4. Training Configuration

| Parameter | Value |
|---|---|
| Random seed | 42 |
| Batch size | 16 |
| Optimizer | Adam |
| Initial learning rate | 0.001 |
| Maximum epochs | 200 |
| Minimum epochs | 30 |
| Early stopping patience | 20 |
| Minimum improvement | 0.001 |
| LR scheduler | ReduceLROnPlateau |
| Scheduler factor | 0.5 |
| Scheduler patience | 5 |
| Minimum learning rate | 0.00001 |
| Loss | Log-magnitude L1 |
| Checkpoint selection | Validation PESQ |

## 5. Training Results

| Item | Result |
|---|---:|
| Best epoch | 90 |
| Early stopping epoch | 110 |
| Best validation PESQ | 1.7537 |
| Metric mode | PESQ |
| Training time | 2,945.14 seconds |
| GPU | NVIDIA Tesla T4 |

The best checkpoint was selected at epoch 90.
Training terminated at epoch 110 through Early Stopping.

## 6. Validation Evaluation

The model was evaluated on all 164 Validation audio pairs.

| Metric | Noisy | Enhanced | Improvement |
|---|---:|---:|---:|
| PESQ | 1.3947 | 1.7537 | +0.3591 |
| SNR (dB) | 9.6341 | 13.6130 | +3.9789 dB |

### Evaluation Summary

- Mean Noisy PESQ: 1.3947
- Mean Enhanced PESQ: 1.7537
- PESQ improvement: +0.3591
- Mean Noisy SNR: 9.6341 dB
- Mean Enhanced SNR: 13.6130 dB
- SNR improvement: +3.9789 dB
- Validation sample count: 164

## 7. Files

- Model checkpoint: `results/baseline/best_model.pt`
- Training log: `results/baseline/training_log.csv`
- Evaluation CSV: `results/baseline/evaluation/evaluation.csv`
- Evaluation summary: `results/baseline/evaluation/summary.json`

## 8. Interpretation

The Baseline model improved both PESQ and SNR on the Validation dataset.

PESQ increased by 0.3591, while SNR increased by approximately 3.9789 dB.
These results indicate that the shared MaskCNN baseline achieved measurable improvements in perceptual speech quality and signal-to-noise ratio.

However, the Baseline is not considered the final model.
It serves as the common reference point for subsequent team experiments.

Future experiments should use the same Validation dataset and evaluation pipeline whenever possible to ensure fair comparison.

## 9. Team Experiment Guidelines

1. Use the shared Baseline as the reference model.
2. Conduct independent experiments using different architectures, loss functions, preprocessing methods, or training strategies.
3. Record experiment configurations and results.
4. Evaluate using the shared evaluation pipeline.
5. Prioritize Validation PESQ-WB when comparing models.
6. Use SNR as a supplementary metric.
7. Avoid modifying the shared evaluation procedure without documenting the changes.
