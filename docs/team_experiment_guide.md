%%writefile /content/drive/MyDrive/signals_and_systems_v2/team_repo/docs/team_experiment_guide.md

# 팀원 실험 가이드

## 1. 프로젝트 목표

공통 Baseline 모델을 기반으로 전통적인 방식이 아닌 딥러닝 기반 음성 잡음 제거 모델의 성능을 개선한다.

최종 모델은 Validation 데이터 기준 평균 PESQ-WB를 주요 평가 지표로 선정하며, SNR을 보조 지표로 활용한다.

모든 실험은 공통 Baseline과 동일한 데이터 및 평가 조건을 기준으로 비교한다.

---

## 2. 공유 Baseline

### Dataset

- Sample Rate: 16,000 Hz
- Audio Length: 3 seconds
- Samples per audio: 48,000
- Train: 1,296 pairs
- Validation: 164 pairs

### Model

- Model: MaskCNN
- Input: Log-magnitude STFT
- FFT Size: 512
- Hop Length: 128
- Output: Time-frequency mask
- Loss: Log-magnitude L1

### Baseline 성능

| Metric | Noisy | Baseline |
|---|---:|---:|
| PESQ-WB | 1.3947 | 1.7537 |
| SNR (dB) | 9.6341 | 13.6130 |

---

## 3. 실험 원칙

1. 모든 실험은 공유 Baseline을 기준으로 진행한다.
2. Baseline 코드를 직접 덮어쓰지 않는다.
3. 한 실험에서 변경 사항을 명확히 구분하고 기록한다.
4. Train 데이터와 Validation 데이터의 역할을 구분한다.
5. Validation 데이터를 학습 데이터로 사용하거나 Validation 정답을 직접 학습에 활용하지 않는다.
6. Validation 결과를 이용한 모델 선택 및 하이퍼파라미터 조정은 허용한다.
7. Test 데이터는 최종 평가 전까지 사용하지 않는다.
8. 실험이 실패하거나 성능이 하락해도 기록한다.
9. 실험 결과는 동일한 평가 방식으로 측정한다.
10. 결과를 제출할 때 실험 코드와 결과 기록을 함께 제출한다.

---

## 4. 실험 아이디어

다음은 탐색 가능한 실험 범주이다. 특정 범주에 제한되지 않으며 자유롭게 새로운 아이디어를 제안할 수 있다.

### Model Architecture

- CNN 구조 변경
- Layer 수 조정
- Channel 수 조정
- Residual connection
- 다양한 CNN 구조
- 시간적 문맥 정보 활용

### Input Representation

- STFT 파라미터 조정
- FFT Size 변경
- Hop Length 변경
- Multi-resolution STFT
- 입력 특징 변경

### Loss Function

- Magnitude Loss
- Spectral Loss
- Multi-resolution STFT Loss
- Waveform Loss
- Loss 조합

### Data Processing

- Noise augmentation
- 음성 데이터 증강
- Noise 특성별 학습
- 데이터 전처리 변경

### Training Strategy

- Optimizer 변경
- Learning Rate 변경
- Batch Size 변경
- Scheduler 변경
- 학습 안정화 전략

### 기타

위 범주에 포함되지 않는 새로운 방법도 자유롭게 시도할 수 있다.

---

## 5. 실험 진행 절차

### Step 1. 실험 아이디어 선정

개선하고자 하는 문제와 실험 가설을 설정한다.

### Step 2. 실험 환경 확인

공통 환경 및 데이터 경로를 확인한다.

### Step 3. 실험 코드 작성

공통 Baseline 코드를 보존하고 별도의 실험 코드 또는 설정 파일을 작성한다.

### Step 4. 학습

Train 데이터를 이용하여 모델을 학습한다.

### Step 5. Validation 평가

Validation 164개 데이터를 대상으로 PESQ-WB와 SNR을 평가한다.

### Step 6. 결과 분석

Baseline과 비교하고 성능 변화 및 음성 품질을 분석한다.

### Step 7. 실험 기록 작성

`docs/team_experiment_template.md`를 참고하여 실험 기록을 작성한다.

### Step 8. 결과 제출

실험 코드, 설정, 평가 결과, 실험 기록을 제출한다.

---

## 6. Git 관리 규칙

### 실험 브랜치

각 팀원은 개인 실험 브랜치를 사용한다.

브랜치 이름 예시:

- `exp/member1-multiresolution`
- `exp/member2-loss`
- `exp/member3-residual`
- `exp/member4-augmentation`

### Baseline 보호

- `main` 브랜치의 Baseline 코드를 직접 수정하지 않는다.
- 개인 실험은 개인 브랜치에서 수행한다.
- 공통 파일 수정이 필요하면 변경 사항을 명확히 기록한다.
- 실험 결과가 검증되기 전에는 공통 Baseline에 통합하지 않는다.

### 실험 파일 관리

실험 관련 코드는 다음 경로를 활용한다.

- `src/`: 공통 모델 및 공통 기능
- `configs/`: 공통 설정
- `experiments/`: 개인 실험 코드 및 설정
- `docs/`: 실험 기록 및 문서
- `results/`: 실험 평가 결과

개인 실험 코드와 설정은 가능한 한 기존 공통 코드와 분리한다.

### 체크포인트 관리

모델 체크포인트 파일은 용량이 크므로 GitHub에 직접 업로드하지 않는 것을 원칙으로 한다.

필요한 경우 Google Drive 등 별도의 공유 저장 공간에 저장하고 경로 및 관리 방법을 기록한다.

---

## 7. 실험 결과 제출 기준

모든 실험은 다음 항목을 포함해야 한다.

- 실험 목적 및 가설
- 변경 사항
- 학습 설정
- Validation PESQ-WB
- Validation SNR
- Baseline 대비 성능 변화
- 결과 분석
- 한계점
- 실험 코드
- 재현에 필요한 설정 파일

---

## 8. 모델 채택 기준

최종 모델은 Validation 평균 PESQ-WB를 주요 기준으로 선정한다.

단, PESQ만으로 모델을 판단하지 않고 다음 사항도 검토한다.

- SNR 변화
- 음성 왜곡
- 잔존 잡음
- 음성 자연스러움
- 학습 안정성
- 추론 속도
- 재현 가능성

실험 결과가 Baseline보다 높더라도 음성 품질이나 안정성에 문제가 있다면 추가 검증한다.

---

## 9. 실험 제출 및 통합

각 실험자는 개인 브랜치에서 실험을 완료한 후 결과를 공유한다.

최종 통합은 팀원들의 실험 결과를 비교하고 검토한 후 진행한다.

성능이 우수한 모델을 중심으로 추가 실험을 진행하며, 필요하면 서로 다른 실험 방법을 결합하여 새로운 모델을 개발할 수 있다.
