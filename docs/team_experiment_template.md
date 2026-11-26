# 팀 실험 기록 템플릿

## 1. 실험 기본 정보

- 실험자:
- 실험 ID:
- 실험 날짜:
- 실험 이름:
- 기반 모델: Shared Baseline MaskCNN
- Baseline 체크포인트: `results/baseline/best_model.pt`

## 2. 실험 목적 및 가설

### 실험 목적
이번 실험에서 개선하고자 하는 문제를 작성한다.

### 가설
변경한 방법이 음성 잡음 제거 성능에 어떤 영향을 줄 것이라고 예상하는지 작성한다.

## 3. 변경 사항

| 항목 | Baseline | 실험 모델 |
|---|---|---|
| 모델 구조 | MaskCNN | |
| 입력 특징 | Log-magnitude STFT | |
| Loss | Log-magnitude L1 | |
| FFT | 512 | |
| Hop Length | 128 | |
| Optimizer | Adam | |
| Learning Rate | 0.001 | |
| Batch Size | 16 | |
| 기타 변경 사항 | - | |

## 4. 실험 환경

- Python:
- PyTorch:
- GPU:
- Random Seed:
- 학습 데이터:
- Validation 데이터:

## 5. 학습 설정

- 최대 Epoch:
- Early Stopping:
- Scheduler:
- 학습 시간:
- 기타 설정:

## 6. 실험 결과

| 평가 지표 | Noisy | Baseline | 실험 모델 |
|---|---:|---:|---:|
| 평균 PESQ-WB | | 1.7537 | |
| 평균 SNR (dB) | | 13.6130 | |

### Baseline 대비 변화

- PESQ 변화:
- SNR 변화:

## 7. 결과 분석

### 성능 변화
결과가 개선되었는지, 악화되었는지 분석한다.

### 음성 품질 분석
음성 왜곡, 잡음 잔존, 음성 선명도 등을 분석한다.

### 가설 검증
초기 가설이 실제 결과와 일치했는지 작성한다.

### 한계점
실험에서 발견한 문제나 추가 개선이 필요한 부분을 작성한다.

## 8. 실험 산출물

- 학습 코드:
- 평가 코드:
- 모델 체크포인트:
- 평가 CSV:
- 기타 산출물:

## 9. 최종 결론

- 실험 채택 여부:
- 채택 / 보류 / 기각
- 판단 근거:
- 후속 실험:
