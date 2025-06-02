import json
import logging
from models.course import Course, TimeSlot # TimeSlot 클래스도 임포트

class BaseRecommender:
    """
    모든 추천 시스템의 기본 클래스입니다.
    과목 로딩 및 공통 유틸리티 함수를 포함합니다.
    """
    def __init__(self):
        self.courses = []
        self.course_map = {} # 과목 코드를 키로 Course 객체를 빠르게 찾기 위한 맵

    def load_courses(self, filename):
        """timetable.json 파일에서 과목 데이터를 로드합니다."""
        self.courses = []
        self.course_map = {}
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except FileNotFoundError:
            logging.error(f"ERROR: '{filename}' 파일을 찾을 수 없습니다.")
            return False
        except json.JSONDecodeError:
            logging.error(f"ERROR: '{filename}' JSON 파일 디코딩 오류. 파일 형식을 확인해주세요.")
            return False

        for item in data:
            try:
                # Course.from_dict를 사용하여 Course 객체 생성
                # Course.from_dict는 내부적으로 TimeSlot.from_dict를 호출하여 times 리스트를 올바르게 변환합니다.
                course = Course.from_dict(item) 
                
                self.courses.append(course)
                self.course_map[course.code] = course
            except KeyError as e:
                logging.error(f"ERROR: JSON에서 과목 로드 중 오류 (누락된 키: {e}): {item}")
                return False
            except TypeError as e:
                logging.error(f"ERROR: JSON에서 과목 로드 중 타입 오류: {e} in {item}")
                return False
            except Exception as e: # TimeSlot 변환 오류 등 기타 오류 포착
                logging.error(f"ERROR: JSON에서 과목 로드 중 알 수 없는 오류: {e} in {item}")
                return False

        logging.info(f"INFO: {len(self.courses)}개의 과목을 '{filename}'에서 성공적으로 로드했습니다.")
        return True

    def calculate_total_credits(self, schedule_courses):
        """주어진 시간표의 총 학점을 계산합니다."""
        return sum(course.credits for course in schedule_courses)

    def is_time_conflict(self, schedule_courses, new_course=None):
        """
        주어진 시간표 내에서 시간 충돌이 있는지 확인합니다.
        새로운 과목이 기존 시간표와 충돌하는지 확인할 때도 사용됩니다.
        """
        all_time_slots = []
        for course in schedule_courses:
            all_time_slots.extend(course.times)
        
        if new_course:
            all_time_slots.extend(new_course.times)

        for i in range(len(all_time_slots)):
            for j in range(i + 1, len(all_time_slots)):
                ts1 = all_time_slots[i]
                ts2 = all_time_slots[j]

                # TimeSlot 객체의 속성에 접근
                if ts1.day == ts2.day: # .day로 접근
                    if max(ts1.start, ts2.start) < min(ts1.end, ts2.end): # .start, .end로 접근
                        return True # 충돌 발생
        return False # 충돌 없음

    def generate_timetable_html(self, schedule_courses, title="추천 시간표"):
        """주어진 시간표를 HTML 테이블 형식으로 변환합니다."""
        logging.info(f"INFO: generate_timetable_html 호출됨. 받은 스케줄 과목 수: {len(schedule_courses)}")
        
        if not schedule_courses:
            return f"""
                <h3>{title}</h3>
                <p>추천된 과목이 없습니다. 조건을 완화하거나 과목 데이터를 확인해주세요.</p>
                <p>총 학점: 0 / 총 과목 수: 0</p>
            """

        hours = range(9, 18) # 9시부터 17시까지 (18시 시작 수업은 미표기)
        days = ["월", "화", "수", "목", "금"] # 0-4에 대응

        # 시간표 그리드 초기화
        # timetable_grid[시간][요일_인덱스] = 과목 정보 또는 "covered"
        timetable_grid = {hour: {day_idx: None for day_idx in range(5)} for hour in hours}
        
        # 과목별 색상 할당
        subject_colors = {}
        color_palette = [
            "#FFD700", "#FF6347", "#6A5ACD", "#3CB371", "#FF8C00",
            "#4682B4", "#DC143C", "#ADFF2F", "#BA55D3", "#F08080",
            "#20B2AA", "#8A2BE2", "#5F9EA0", "#D2691E", "#FF4500",
            "#87CEEB", "#FFB6C1", "#98FB98", "#DDA0DD", "#FFE4B5"
        ]
        color_index = 0

        for course in schedule_courses:
            if course.name not in subject_colors:
                subject_colors[course.name] = color_palette[color_index % len(color_palette)]
                color_index += 1

            for time_slot in course.times:
                # TimeSlot 객체의 속성에 접근
                day_index = time_slot.day
                start_time_minutes = time_slot.start
                end_time_minutes = time_slot.end
                
                # 시작 시간의 시간 블록 (예: 900 -> 9시)
                start_hour_block = start_time_minutes // 100
                # 끝 시간의 시간 블록 (예: 1000 -> 10시, 실제로는 9시 블록까지만 차지)
                # end_time_minutes가 1000이면 10:00에 끝나는 것이므로 9시 블록까지
                end_hour_block = (end_time_minutes - 1) // 100 
                
                # colspan이 아니라 rowspan
                calculated_rowspan = end_hour_block - start_hour_block + 1

                # 해당 시간 블록과 요일 인덱스가 유효한 범위 내에 있을 경우
                if start_hour_block in hours and day_index in range(5):
                    # 이미 해당 셀이 'covered'이거나 다른 과목으로 채워져 있는지 확인
                    if timetable_grid[start_hour_block][day_index] is None:
                        timetable_grid[start_hour_block][day_index] = {
                            "subject": course.name,
                            "location": course.location,
                            "credits": course.credits,
                            "professor": course.professor,
                            "rowspan": calculated_rowspan,
                            "color": subject_colors.get(course.name, "#4caf50")
                        }
                    else:
                        logging.warning(f"WARNING: 시간표 그리드에 이미 항목이 존재: {course.name} at {start_hour_block}:00 {days[day_index]}")

                    # 수업이 차지하는 다음 시간 블록들을 "covered"로 표시하여 중복 렌더링 방지
                    for hour_to_cover in range(start_hour_block + 1, start_hour_block + calculated_rowspan):
                        if hour_to_cover in hours and day_index in range(5):
                            # 이미 다른 과목이 시작하는 셀이 아닌 경우에만 'covered'로 표시
                            if timetable_grid[hour_to_cover][day_index] is None:
                                timetable_grid[hour_to_cover][day_index] = "covered"
                            else:
                                logging.warning(f"WARNING: 'covered'로 설정하려 했으나 이미 항목이 존재: {hour_to_cover}:00 {days[day_index]}")
                else:
                    logging.warning(f"WARNING: 과목 '{course.name}'의 시간표 슬롯이 정의된 범위({hours.start}-{hours.stop}시, 월-금) 밖입니다: 요일={day_index}, 시작={start_hour_block}")


        html_content = f'<h3>{title}</h3>'
        html_content += '<table class="timetable">'
        html_content += '<thead>'
        html_content += '<tr>'
        html_content += '<th>시간/요일</th>'
        for day_name in days:
            html_content += f"<th>{day_name}</th>\n"
        html_content += '</tr>'
        html_content += '</thead>'
        html_content += '<tbody>'

        for hour in hours:
            html_content += f"""<tr>
                <td>{hour:02d}:00</td>
            """
            for day_index in range(5):
                cell_data = timetable_grid[hour][day_index]

                if cell_data == "covered":
                    continue # 이미 이전 셀에서 rowspan으로 커버된 셀은 건너뜀
                elif cell_data:
                    subject = cell_data["subject"]
                    color = cell_data["color"]
                    rowspan = cell_data["rowspan"]
                    location = cell_data["location"]
                    credits = cell_data["credits"]
                    professor = cell_data["professor"]
                    
                    html_content += f"""<td rowspan="{rowspan}" style="background-color:{color};">
                        <div class="class-block">
                            <span>{subject}</span>
                            <span>{professor}</span>
                            <span>{location}</span>
                            <span>{credits}학점</span>
                        </div>
                    </td>
                    """
                else:
                    html_content += "<td></td>\n" # 비어있는 셀
            html_content += "</tr>\n"

        html_content += """
            </tbody>
        </table>
        """
        
        total_credits = self.calculate_total_credits(schedule_courses)
        total_courses_count = len(schedule_courses) # 변수명 변경 (total_courses와의 충돌 방지)

        html_content += '<p><strong>포함된 과목:</strong></p>'
        html_content += '<ul class="course-list">'
        for course in schedule_courses:
            html_content += f'<li><span class="course-name">{course.name} ({course.code})</span> <span class="course-details">{course.credits}학점 / {course.professor} / {course.location}</span></li>'
        html_content += '</ul>'
        
        html_content += f'<p>총 학점: {total_credits} / 총 과목 수: {total_courses_count}</p>'

        return html_content

    def recommend_schedule(self, user_prefs):
        """
        이 메서드는 하위 클래스에서 구현되어야 합니다.
        """
        raise NotImplementedError("recommend_schedule 메서드는 하위 클래스에서 구현되어야 합니다.")