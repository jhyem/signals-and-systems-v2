
# 팀 실험 Git 운영 가이드

## 1. 기본 원칙

- main 브랜치는 공통 Baseline 및 검증된 코드 유지용이다.
- 개인 실험은 반드시 개인 실험 브랜치에서 수행한다.
- main 브랜치에 직접 실험 코드를 수정하거나 푸시하지 않는다.
- 실험 브랜치에서 변경한 코드와 결과를 제출한다.
- 최종 통합은 실험 결과 검토 후 진행한다.

## 2. 실험 브랜치 생성

GitHub 저장소를 Clone한 후 main을 기준으로 실험 브랜치를 생성한다.

```bash
git clone https://github.com/jhyem/signals-and-systems-v2.git
cd signals-and-systems-v2

git checkout main
git pull origin main
git checkout -b exp/member1-experiment-name
