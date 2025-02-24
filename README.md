#품질 검사 자동화 시스템 🏭

## 프로젝트 소개

이 프로젝트는 철강 제품의 품질 검사를 자동화하는 AI 기반 시스템입니다. 딥러닝을 활용하여 제품의 결함을 자동으로 검출하고 측정하며, 최종적으로 품질 등급을 판정합니다.

## 시스템 작동 방식
![시스템 흐름도]

### 1단계: 이미지 입력 및 객체 검출 🔍
- YOLO 모델이 이미지에서 두 가지를 찾습니다:
  1. 제품 식별 정보가 있는 Bar 영역
  2. 결함이 있는 Overfill 영역

### 2단계: 텍스트 인식 및 처리 📝
- Bar 영역에서:
  1. 텍스트 방향 감지 및 보정
  2. OCR로 텍스트 추출
  3. Heat Number와 ID Number 식별

### 3단계: 결함 분석 ⚖️
- Overfill 영역에서:
  1. 정밀한 영역 분할(세그멘테이션)
  2. 결함 면적 계산
  3. 결함 길이 측정

### 4단계: 품질 등급 판정 📊
- 결함 분석 결과에 따라 등급 부여:
  - A등급: 결함 없음
  - C등급: 결함 있음
  - D등급: ID 인식 불가

## 학습 과정에서의 도전과 해결

### 1. 데이터 불균형 문제
처음에는 Bar와 Overfill 데이터의 큰 불균형(200:40)으로 어려움을 겪었습니다.

**해결 방법:**
- Overfill 클래스에 가중치 부여
- 데이터 증강 기법 적용
- 추가 데이터 수집

### 2. 정밀한 결함 측정
결함 영역의 정확한 측정이 필요했습니다.

**해결 방법:**
```python
# 마스크를 YOLO 형식으로 변환
contours = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
for cnt in contours:
    if cv2.contourArea(cnt) > 200:  # 노이즈 제거
        polygon = normalize_coordinates(cnt)
        polygons.append(polygon)
```

### 3. 텍스트 인식 정확도
회전된 텍스트나 품질이 좋지 않은 텍스트의 인식이 어려웠습니다.

**해결 방법:**
- 텍스트 방향 자동 보정
- Heat Number 규칙 기반 처리
- ID Number 정규화

## 시스템 구성 요소

### 🤖 사용된 AI 모델
1. **YOLOv8**
   - 객체 검출 및 세그멘테이션
   - 1000 에포크 학습
   - mAP50: 0.95 이상 달성

2. **PaddleOCR**
   - 텍스트 박스 검출
   - 문자 인식
   - 방향 보정 기능

### 📁 프로젝트 구조
```
project/
├── 📄 main.py          # 메인 프로그램
├── 🖥️ main_ui.py       # GUI 프로그램
├── 📸 img_rotation.py  # 이미지 처리
├── 🔍 overfill_detection.py
├── 📏 overfill_measurement.py
└── 🎯 segment.py
```

## 사용 방법

### 1. 설치하기
```bash
git clone [repository-url]
cd project-directory
pip install -r requirements.txt
```

### 2. 실행하기
```bash
# GUI 모드
python main_ui.py

# 커맨드 라인 모드
python main.py --input_dir /path/to/images
```

## 성능 향상 과정

처음 시작할 때와 비교한 성능 향상:
```
객체 검출: mAP50 0.82 → 0.95
세그멘테이션: IoU 0.78 → 0.85
측정 오차: ±8% → ±5%
```

## 🔜 앞으로의 계획
1. 실시간 처리 속도 개선
2. 추가 결함 유형 탐지
3. 웹 인터페이스 개발
4. 모델 경량화

## 팀원 소개
Besteel Juniors 팀이 개발했습니다.

---
※ 본 프로젝트는 SeAH Besteel의 철강 제품 품질 관리 프로세스 개선을 위해 개발되었습니다.
