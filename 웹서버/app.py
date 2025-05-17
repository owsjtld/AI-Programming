from flask import Flask, render_template, request
from timetable import ScheduleRecommender, UserPreferences
import pandas as pd
import os

app = Flask(__name__)

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    # 1) 폼 입력값 파싱
    min_credits = int(request.form['min_credits'])
    max_credits = int(request.form['max_credits'])
    preferred_days = [
        int(d.strip()) for d in request.form.get('preferred_days', '').split(',') if d.strip().isdigit()
    ]
    preferred_professors = [
        p.strip() for p in request.form.get('preferred_professors', '').split(',') if p.strip()
    ]
    excluded_courses = [
        c.strip() for c in request.form.get('excluded_courses', '').split(',') if c.strip()
    ]
    preferred_difficulty = float(request.form['preferred_difficulty'])
    preferred_rating = float(request.form['preferred_rating'])

    prefs = UserPreferences(
        min_credits=min_credits,
        max_credits=max_credits,
        preferred_days=preferred_days,
        preferred_professors=preferred_professors,
        excluded_courses=excluded_courses,
        preferred_difficulty=preferred_difficulty,
        preferred_rating=preferred_rating
    )

    # 2) Recommend 시스템 초기화 및 데이터 로드
    rec = ScheduleRecommender()
    rec.load_courses_from_json('timetable.json')

    # 3) (선택) AI 학습 데이터가 있으면 모델 훈련
    if os.path.exists('training_data.csv'):
        df = pd.read_csv('training_data.csv')
        rec.train_model(df)

    # 4) 사용자 선호도 설정 및 추천
    rec.set_user_preferences(prefs)
    recommendations = rec.generate_recommendations()

    if not recommendations:
        return "<h2>조건에 맞는 시간표를 찾을 수 없습니다.</h2><a href='/'>다시</a>"

    best_schedule, best_score = recommendations[0]

    # 5) HTML 생성 및 렌더링
    timetable_html = rec.generate_html(best_schedule, best_score)
    return render_template('result.html', timetable_html=timetable_html)

if __name__ == '__main__':
    app.run(debug=True)
