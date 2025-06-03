import logging
import traceback # traceback 모듈 임포트 추가
from flask import Flask, render_template, request, jsonify, url_for
from models.user_preferences import UserPreferences
from recommender.sarsa_recommender import SarsaRecommender
import json

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

app = Flask(__name__)
recommender = SarsaRecommender()

@app.before_request
def load_data_before_request():
    """요청 처리 전에 과목 데이터를 로드합니다."""
    if not recommender.courses: 
        logging.info("INFO: 애플리케이션 시작 전/첫 요청 시 과목 데이터 로딩 시도.")
        if not recommender.load_courses('timetable.json'):
            logging.error("ERROR: timetable.json 로드 실패. 애플리케이션이 올바르게 작동하지 않을 수 있습니다.")

@app.route('/')
def index():
    """메인 시간표 추천 입력 페이지를 렌더링합니다."""
    return render_template('insert.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    """사용자 선호도를 받아 시간표를 추천하고 결과를 보여줍니다."""
    try:
        min_credits = int(request.form['min_credits'])
        max_credits = int(request.form['max_credits'])
        
        preferred_days = [int(d) for d in request.form.getlist('preferred_days') if d.strip()]
        excluded_days = [int(d) for d in request.form.getlist('excluded_days') if d.strip()]
        
        excluded_courses_str = request.form.get('excluded_courses', '')
        excluded_courses = [c.strip() for c in excluded_courses_str.split(',') if c.strip()]
        
        preferred_rating_str = request.form.get('preferred_rating', '3.5')
        preferred_rating = float(preferred_rating_str) if preferred_rating_str else 3.5
        
        gap_preference_level = request.form.get('gap_preference_level', 'medium') 

        preferred_grade_str = request.form.get('preferred_grade', '0')
        preferred_grade = int(preferred_grade_str) if preferred_grade_str else 0

        user_prefs = UserPreferences(
            min_credits=min_credits,
            max_credits=max_credits,
            preferred_days=preferred_days,
            excluded_days=excluded_days,
            excluded_courses=excluded_courses,
            preferred_rating=preferred_rating,
            gap_preference_level=gap_preference_level,
            preferred_grade=preferred_grade
        )
        
        logging.info(f"INFO: 사용자 선호도 수신: {user_prefs.to_dict()}")

        recommended_schedules_list = recommender.recommend_schedule(user_prefs, num_recommendations=3, num_episodes=20000)
        
        logging.info(f"INFO: {len(recommended_schedules_list)}개의 시간표가 추천되었습니다.")

        timetables_for_display = []
        for i, schedule_courses in enumerate(recommended_schedules_list):
            html_content = recommender.generate_timetable_html(schedule_courses, title=f"추천 시간표 {i+1}")
            
            total_credits = recommender.calculate_total_credits(schedule_courses)
            total_courses_count = len(schedule_courses)
            
            # 각 과목의 상세 정보를 포함하는 리스트 생성 (학년 정보 포함)
            detailed_courses_info = []
            for course in schedule_courses:
                detailed_courses_info.append({
                    'code': course.code,
                    'name': course.name,
                    'credits': course.credits,
                    'grade': course.grade,
                    'professor': course.professor,
                    'times': [ts.to_dict() for ts in course.times], 
                    'location': course.location, # <-- 'room' 대신 'location'으로 변경
                    'rating': course.rating
                })

            timetables_for_display.append({
                'id': i + 1,
                'html': html_content,
                'courses_data_for_view': [c.code for c in schedule_courses], 
                'total_credits': total_credits, 
                'total_courses_count': total_courses_count,
                'detailed_courses': detailed_courses_info # 추가된 상세 과목 정보
            })

        return render_template('result.html', recommended_timetables=timetables_for_display)

    except Exception as e:
        logging.error(f"ERROR: 시간표 추천 중 오류 발생: {e}", exc_info=True)
        return render_template('insert.html', error_message=f"오류 발생: {e}. 입력 값을 확인하고 다시 시도해 주세요.")


@app.route('/view_schedule/<string:schedule_courses_str>')
def view_schedule(schedule_courses_str):
    """선택된 시간표의 상세 정보를 보여줍니다."""
    try:
        selected_course_codes = schedule_courses_str.split('-')
        
        selected_courses = []
        detailed_courses_info = [] # 상세 정보를 담을 리스트 추가
        for code in selected_course_codes:
            course = recommender.get_course_by_code(code)
            if course:
                selected_courses.append(course)
                detailed_courses_info.append({ # 상세 정보 추가
                    'code': course.code,
                    'name': course.name,
                    'credits': course.credits,
                    'grade': course.grade,
                    'professor': course.professor,
                    'times': [ts.to_dict() for ts in course.times],
                    'location': course.location, # <-- 'room' 대신 'location'으로 변경
                    'rating': course.rating
                })
            else:
                logging.warning(f"WARNING: 찾을 수 없는 과목 코드: {code} (view_schedule)")
        
        schedule_html = recommender.generate_timetable_html(selected_courses, title="선택된 시간표")
        
        return render_template('selected_schedule.html', 
                               single_schedule_html=schedule_html,
                               detailed_courses=detailed_courses_info) # 상세 과목 정보 전달

    except Exception as e:
        logging.error(f"ERROR: 상세 시간표 보기 중 오류 발생: {e}", exc_info=True)
        return render_template('insert.html', error_message=f"상세 시간표 보기 중 오류가 발생했습니다: {e}. 다시 시도해 주세요.")

if __name__ == '__main__':
    app.run(debug=True)
