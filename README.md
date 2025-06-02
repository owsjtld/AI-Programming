---

## 📅 SARSA 기반 학점 시간표 추천 시스템

이 프로젝트는 강화 학습(Reinforcement Learning)의 SARSA 알고리즘을 활용하여 사용자 맞춤형 학점 시간표를 추천해 주는 웹 애플리케이션입니다. 사용자의 선호 학점, 요일, 제외 과목 등 다양한 조건을 고려하여 최적의 시간표를 찾아줍니다.

---

### 🌟 주요 기능

* **맞춤형 시간표 추천**: 최소/최대 학점, 선호 요일, 제외 과목, 선호 과목 평점, 공강 최소화 등 사용자의 상세한 선호도를 반영하여 시간표를 생성합니다.
* **강화 학습 기반 최적화**: SARSA 알고리즘을 통해 수많은 에피소드를 반복하며 사용자 선호도에 가장 적합한 시간표 조합을 학습하고 추천합니다.
* **시각적 시간표 제공**: 추천된 시간표를 직관적인 HTML 테이블 형태로 시각화하여 한눈에 파악할 수 있도록 돕습니다.
* **과목 정보 관리**: 과목 코드, 이름, 교수, 학점, 시간, 평점, 위치, 난이도, 학년 등 상세한 과목 정보를 효율적으로 관리합니다.

---

### 🛠️ 기술 스택

* **백엔드**: Python, Flask
* **강화 학습**: SARSA 알고리즘
* **데이터 모델링**: Python Classes (`Course`, `TimeSlot`, `UserPreferences`)
* **프론트엔드**: HTML, CSS, JavaScript (기본적인 웹 인터페이스)
* **데이터 형식**: JSON

---

### 🚀 설치 및 실행 방법

이 프로젝트를 로컬 환경에서 설치하고 실행하는 방법은 다음과 같습니다.

1.  **리포지토리 클론**:
    ```bash
    git clone [Your Repository URL]
    cd [Your Repository Name]
    ```

2.  **가상 환경 설정 (권장)**:
    ```bash
    python -m venv venv
    # Windows
    .\venv\Scripts\activate
    # macOS/Linux
    source venv/bin/activate
    ```

3.  **필요한 라이브러리 설치**:
    프로젝트에 필요한 모든 Python 라이브러리는 `requirements.txt` 파일에 명시되어 있습니다.
    ```bash
    pip install -r requirements.txt
    ```
    (만약 `requirements.txt` 파일이 없다면, `Flask`를 수동으로 설치해야 합니다: `pip install Flask`)

4.  **데이터 파일 준비**:
    프로젝트 루트 디렉토리에 `timetable.json` 파일이 존재하는지 확인해주세요. 이 파일에는 시스템이 추천할 과목들의 데이터가 JSON 형식으로 포함되어 있어야 합니다. 예시 형식은 다음과 같습니다:
    ```json
    [
        {
            "code": "CS101",
            "name": "자료구조",
            "professor": "김교수",
            "credits": 3,
            "times": [
                {"day": 0, "start": 900, "end": 950},
                {"day": 2, "start": 900, "end": 950}
            ],
            "rating": 4.2,
            "location": "공학관 101호",
            "difficulty": 0.7,
            "grade": 1
        },
        {
            "code": "MA201",
            "name": "선형대수",
            "professor": "이교수",
            "credits": 3,
            "times": [
                {"day": 1, "start": 1000, "end": 1150}
            ],
            "rating": 3.8,
            "location": "자연대 203호",
            "difficulty": 0.8,
            "grade": 2
        }
    ]
    ```

5.  **애플리케이션 실행**:
    ```bash
    python main.py
    ```

6.  **웹 브라우저 접근**:
    애플리케이션이 실행되면, 웹 브라우저를 열고 다음 주소로 접속합니다:
    ```
    [http://127.0.0.1:5000/](http://127.0.0.1:5000/)
    ```
    이후 웹 인터페이스를 통해 학점 시간표 추천 기능을 사용할 수 있습니다.

---

### 📁 코드 구조

프로젝트의 주요 파일 및 디렉토리 구조는 다음과 같습니다:
.
├── main.py
├── timetable.json
├── templates/
│   ├── error.html
│   ├── insert.html
│   ├── result.html
│   └── selected_schedule.html
├── models/
│   ├── init.py
│   ├── course.py
│   └── user_preferences.py
└── recommender/
├── init.py
├── base_recommender.py
└── sarsa_recommender.py

* **`main.py`**: Flask 웹 애플리케이션의 메인 엔트리 포인트입니다. 사용자 요청 처리, 추천 시스템 연동, HTML 템플릿 렌더링을 담당합니다.
* **`timetable.json`**: 모든 과목 정보가 저장된 JSON 형식의 데이터 파일입니다.
* **`templates/`**: 웹 페이지를 렌더링하는 HTML 템플릿 파일들을 포함합니다.
    * `error.html`: 오류 발생 시 표시되는 페이지.
    * `insert.html`: 사용자가 시간표 선호도를 입력하는 폼 페이지.
    * `result.html`: 추천된 시간표 목록을 보여주는 페이지.
    * `selected_schedule.html`: 특정 추천 시간표의 상세 정보를 보여주는 페이지.
* **`models/`**: 데이터 구조를 정의하는 Python 클래스들을 포함합니다.
    * **`course.py`**: `TimeSlot` (수업 시간)과 `Course` (과목 정보) 클래스를 정의합니다. 각 클래스는 딕셔너리로부터 객체를 생성하고 객체를 딕셔너리로 변환하는 헬퍼 메서드를 포함합니다.
    * **`user_preferences.py`**: `UserPreferences` 클래스를 정의하여 사용자의 시간표 관련 선호도(학점, 요일, 제외 과목 등)를 관리합니다.
* **`recommender/`**: 시간표 추천 로직을 포함하는 모듈입니다.
    * **`base_recommender.py`**: 모든 추천 시스템이 상속받을 기본 클래스입니다. 과목 로딩, 학점 계산, 시간 충돌 확인, HTML 시간표 생성 등 공통 유틸리티 함수를 제공합니다.
    * **`sarsa_recommender.py`**: SARSA 강화 학습 알고리즘을 구현한 핵심 추천 모듈입니다. Q-러닝 테이블을 사용하여 최적의 시간표 조합을 학습하고 예측합니다.

---

### 🧠 SARSA 강화 학습 적용

이 시스템은 **SARSA (State-Action-Reward-State-Action)** 강화 학습 알고리즘을 사용하여 시간표를 추천합니다.

1.  **상태 (State)**: 현재까지 구성된 시간표 (선택된 과목들의 집합)와 사용자 선호도가 결합된 형태로 정의됩니다.
2.  **행동 (Action)**: 현재 상태에서 시간표에 새로운 과목을 추가하는 것을 의미합니다.
3.  **보상 (Reward)**: 새로운 과목이 시간표에 추가되었을 때, 해당 시간표가 사용자 선호도(학점 범위, 선호/제외 요일, 공강 최소화, 과목 평점, 학년 일치 여부 등)에 얼마나 부합하는지에 따라 긍정적 또는 부정적 보상이 주어집니다. 특히, 시간 겹침이나 제외 과목/요일 위반 시에는 큰 페널티를 부여하여 학습을 유도합니다.
4.  **학습**: 에이전트는 수많은 에피소드를 반복하며 Q-테이블을 업데이트합니다. `epsilon-greedy` 전략을 통해 초기에는 다양한 시간표 조합을 탐색하고(탐험), 점차 학습된 Q-값을 바탕으로 최적의 시간표를 선택하는(활용) 방향으로 전환합니다.

이러한 학습 과정을 통해 시스템은 사용자의 복잡한 요구사항을 만족시키면서도 유효한 시간표를 효과적으로 생성할 수 있게 됩니다.

---