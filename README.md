readme = """# 딥러닝 기반 음성 잡음 제거 챌린지

## 1. 프로젝트 개요

본 저장소는 딥러닝 기반 음성 잡음 제거 챌린지를 위한
팀 공통 실험 환경 및 연구 기록 공간입니다.

잡음이 포함된 음성에서 잡음을 제거하고 깨끗한 음성을 복원하는
것을 목표로 딥러닝 및 디지털 신호 처리 기법을 활용합니다.

제공된 Baseline CNN을 공통 기준 모델로 설정하며,
팀원들은 모델 구조, Loss Function, 입력 표현 방식,
학습 전략 및 전통적인 신호 처리 기법과의 결합 등을
자유롭게 탐색하고 실험할 수 있습니다.

최종 모델 후보는 Validation 성능 및 추가 검증 결과를
바탕으로 선정합니다.

## 2. Baseline 모델

| 항목 | 설정 |
|---|---|
| 샘플링 주파수 | 16 kHz |
| 음성 길이 | 3초 |
| 학습 데이터 | 1,296 쌍 |
| 검증 데이터 | 164 쌍 |
| 모델 | MaskCNN (Conv1d) |
| STFT | n_fft=512, hop_length=128 |
| Loss Function | Log-Magnitude L1 |
| Optimizer | Adam |
| Batch Size | 16 |
| 초기 Learning Rate | 0.001 |

### Baseline Validation 결과

| 평가 지표 | 결과 |
|---|---:|
| PESQ-WB | 1.6922 |
| SNR | 13.0604 dB |

Baseline 모델은 모든 실험의 공통 비교 기준으로 사용합니다.

## 3. 실험 규칙

- Baseline 모델을 공통 출발점으로 사용합니다.
- 실험 방법이나 알고리즘을 사전에 제한하지 않습니다.
- 공통 Train 및 Validation 데이터셋을 사용합니다.
- Validation 데이터 164개 전체를 평가합니다.
- 주요 평가 지표는 평균 PESQ-WB입니다.
- SNR 및 음성 청취 결과를 보조 평가에 활용합니다.
- 모델 구조, Loss Function, 학습 방식, 입력 표현 방식 등을 자유롭게 변경할 수 있습니다.
- 전통적인 DSP 기법과 딥러닝의 결합을 허용합니다.
- 실험 코드, 설정, 결과 및 필요한 모델 가중치를 기록합니다.
- Hidden Test 데이터는 모델 선택 및 실험에 사용하지 않습니다.
- 실험 재현성을 고려하여 설정과 실행 환경을 기록합니다.

## 4. 개발 환경

Python 및 PyTorch 기반 환경을 사용합니다.

필요한 Python 패키지는 `requirements.txt`에 정리되어 있습니다.

Google Colab 환경에서 GPU를 활용한 실험을 진행할 수 있습니다.

실험 환경 및 패키지 버전은 실험 결과를 기록할 때 함께 관리합니다.

## 5. 저장소 구조

```text
team_repo/
├── src/                         # 공통 소스 코드
├── configs/                      # 실험 설정 파일
│   └── baseline.json
├── notebooks/                    # 실험 Notebook
├── docs/                         # 실험 문서 및 기록 양식
│   └── team_experiment_template.md
├── requirements.txt               # Python 패키지 목록
├── README.md                      # 프로젝트 안내
└── .gitignore                     # Git 제외 파일 설정
