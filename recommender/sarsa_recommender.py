import random
import json
import logging
from collections import defaultdict
import collections 
from models.course import Course 
from models.user_preferences import UserPreferences 
from recommender.base_recommender import BaseRecommender

class SarsaRecommender(BaseRecommender):
    def __init__(self, alpha=0.1, gamma=0.9, epsilon=1.0, epsilon_decay=0.99995, min_epsilon=0.05):
        super().__init__()
        self.q_table = defaultdict(lambda: defaultdict(float))
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay 
        self.min_epsilon = min_epsilon 
        self.courses_by_code = {}

    def load_courses(self, filepath):
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                courses_data = json.load(f)
            self.courses = [Course.from_dict(data) for data in courses_data]
            self.courses_by_code = {course.code: course for course in self.courses}
            logging.info(f"INFO: {len(self.courses)}개의 과목을 '{filepath}'에서 성공적으로 로드했습니다.")
            return True
        except FileNotFoundError:
            logging.error(f"ERROR: '{filepath}' 파일을 찾을 수 없습니다. 경로를 확인해주세요.")
            return False
        except json.JSONDecodeError as e:
            logging.error(f"ERROR: '{filepath}' JSON 디코딩 오류: {e}")
            return False
        except Exception as e:
            logging.error(f"ERROR: 과목 로드 중 알 수 없는 오류 발생: {e}")
            return False

    def get_course_by_code(self, code):
        return self.courses_by_code.get(code)

    def is_conflict(self, schedule_courses, new_course):
        for existing_course in schedule_courses:
            for new_time in new_course.times:
                for existing_time in existing_course.times:
                    if new_time.day == existing_time.day:
                        if max(new_time.start, existing_time.start) < min(new_time.end, existing_time.end):
                            return True
        return False

    def get_state(self, current_schedule):
        return tuple(sorted([c.code for c in current_schedule]))

    def get_actions(self, current_schedule, user_prefs):
        possible_actions = []
        current_credits = sum(c.credits for c in current_schedule)
        
        for course in self.courses:
            if course in current_schedule:
                continue
            
            if course.code in user_prefs.excluded_courses:
                continue
            
            if current_credits + course.credits > user_prefs.max_credits:
                continue
            
            if self.is_conflict(current_schedule, course):
                continue
            
            is_excluded_day_conflict = False
            for time_slot in course.times:
                if time_slot.day in user_prefs.excluded_days:
                    is_excluded_day_conflict = True
                    break
            if is_excluded_day_conflict:
                continue

            if course.rating < user_prefs.preferred_rating:
                continue

            # --- 학년 필터링 로직 제거 ---
            # 이제 preferred_grade에 따른 과목 제외 로직은 여기에 없습니다.
            # 모든 학년의 과목이 기본적으로 가능한 액션으로 고려됩니다.
            # --- 학년 필터링 로직 끝 ---

            possible_actions.append(course)
        
        return possible_actions

    def calculate_reward(self, schedule, user_prefs, action_course=None, is_terminal=False):
        reward = 0.0
        current_credits = sum(c.credits for c in schedule)

        for course in schedule:
            if course.code in user_prefs.excluded_courses:
                logging.debug(f"DEBUG: 제외 과목 포함 패널티: {course.name}")
                return -float('inf')

        for course in schedule:
            for time_slot in course.times:
                if time_slot.day in user_prefs.excluded_days:
                    logging.debug(f"DEBUG: 제외 요일 충돌 패널티: {course.name} ({time_slot.day}요일)")
                    return -float('inf')

        for course in schedule:
            if course.rating >= user_prefs.preferred_rating:
                reward += 5.0 
            else:
                reward -= 2.0 

        if user_prefs.preferred_days:
            schedule_days = set()
            for course in schedule:
                for time_slot in course.times:
                    schedule_days.add(time_slot.day)
            
            num_preferred_days_in_schedule = len(set(user_prefs.preferred_days).intersection(schedule_days))
            reward += num_preferred_days_in_schedule * 3.0 

        if is_terminal:
            if user_prefs.min_credits <= current_credits <= user_prefs.max_credits:
                target_mid_credits = (user_prefs.min_credits + user_prefs.max_credits) / 2
                reward += (user_prefs.max_credits - abs(current_credits - target_mid_credits)) * 5.0
                logging.debug(f"DEBUG: 학점 적합 보상: {current_credits} (목표: {target_mid_credits})")
                reward += 50.0 
            else:
                reward -= 500.0 
                logging.debug(f"DEBUG: 학점 범위 위반 패널티: {current_credits}학점 (최소 {user_prefs.min_credits}, 최대 {user_prefs.max_credits})")
        else: 
            if current_credits > user_prefs.max_credits:
                reward -= 20.0 

        if action_course:
            reward += 0.5 

        # --- 학년 일치 보상 로직 유지 (페널티 없음) ---
        # 선호 학년이 설정되어 있고, 해당 과목이 선호 학년에 맞으면 추가 보상
        if user_prefs.preferred_grade != 0 and action_course and action_course.grade == user_prefs.preferred_grade:
            reward += 5.0 # 선호 학년 과목에 대한 보상 (가중치)
        # --- 학년 일치 보상 로직 끝 ---

        gap_penalty_per_unit_map = {
            'none': 0.0,
            'low': -0.5, 
            'medium': -1.0, 
            'high': -2.0    
        }
        gap_penalty_per_unit = gap_penalty_per_unit_map.get(user_prefs.gap_preference_level, -1.0)
        
        daily_times = collections.defaultdict(list)
        for course in schedule:
            for ts in course.times:
                daily_times[ts.day].append((ts.start, ts.end))

        for day, time_slots in daily_times.items():
            if not time_slots:
                continue

            time_slots.sort()

            current_end_time = min(ts[0] for ts in time_slots) 

            for start, end in time_slots:
                if start > current_end_time:
                    gap_duration = start - current_end_time 
                    gap_units = gap_duration / 100.0 
                    reward += gap_units * gap_penalty_per_unit 
                
                current_end_time = max(current_end_time, end)
        
        return reward

    def choose_action(self, state, actions):
        if not actions:
            return None
            
        if random.uniform(0, 1) < self.epsilon:
            return random.choice(actions)
        else:
            q_values = {}
            for action in actions:
                action_key = self._get_action_key(action)
                q_values[action] = self.q_table[state][action_key]
            
            if not q_values: 
                return random.choice(actions) 

            max_q = max(q_values.values())
            best_actions = [action for action, q in q_values.items() if q == max_q]
            
            if not best_actions: 
                return random.choice(actions)

            return random.choice(best_actions)

    def _get_action_key(self, action_course):
        return action_course.code

    def learn(self, state, action, reward, next_state, next_action):
        if action is None:
            return

        action_key = self._get_action_key(action)

        next_q = 0.0
        if next_action:
            next_action_key = self._get_action_key(next_action)
            next_q = self.q_table[next_state][next_action_key]
        
        current_q = self.q_table[state][action_key]
        
        self.q_table[state][action_key] += \
            self.alpha * (reward + self.gamma * next_q - current_q)
    
    def recommend_schedule(self, user_prefs: UserPreferences, num_recommendations=3, num_episodes=100000):
        logging.info("INFO: 시간표 추천 로직 (SARSA 강화 학습 아이디어 적용) 시작. (에피소드 수: %d)", num_episodes)
        
        self.epsilon = 1.0 
        self.min_epsilon = 0.05 

        found_schedules_map = {} 
        
        for episode in range(num_episodes):
            current_schedule = []
            state = self.get_state(current_schedule)
            
            possible_actions = self.get_actions(current_schedule, user_prefs)
            action = self.choose_action(state, possible_actions)

            while action is not None:
                next_schedule = current_schedule + [action]
                next_state = self.get_state(next_schedule)
                
                next_possible_actions = self.get_actions(next_schedule, user_prefs)
                next_action = self.choose_action(next_state, next_possible_actions)
                
                reward = self.calculate_reward(next_schedule, user_prefs, action_course=action)

                self.learn(state, action, reward, next_state, next_action)
                
                current_schedule = next_schedule
                state = next_state
                action = next_action

                current_credits = sum(c.credits for c in current_schedule)
                if current_credits >= user_prefs.max_credits or not next_possible_actions:
                    break
                
            final_reward = self.calculate_reward(current_schedule, user_prefs, is_terminal=True)
            
            final_credits = sum(c.credits for c in current_schedule)
            
            credits_ok = (user_prefs.min_credits <= final_credits <= user_prefs.max_credits)
            
            excluded_courses_ok = not any(c.code in user_prefs.excluded_courses for c in current_schedule)
            
            excluded_days_ok = True
            for c in current_schedule:
                for time_slot in c.times:
                    if time_slot.day in user_prefs.excluded_days:
                        excluded_days_ok = False
                        break
                if not excluded_days_ok:
                    break
            
            # --- 학년 일치 여부 확인 (최종 검증 로직은 유지, 보상은 calculate_reward에서) ---
            # preferred_grade_ok는 이제 모든 학년을 허용하므로 항상 True입니다.
            # 하지만 사용자가 특정 학년 과목만 보고 싶어할 수도 있으니, 
            # 이 부분의 로직을 유지하여 최종 추천 시 특정 학년만 포함되도록 할 수도 있습니다.
            # 지금은 모든 학년을 허용하므로, 이 조건을 제거합니다.
            # (만약 '선호 학년'만 보여주되, 학습 시에는 모든 학년 고려를 원한다면 이 부분은 필요)
            # 여기서는 '타 학년 과목도 나오게'라는 요청에 맞춰 이 필터링을 제거합니다.
            
            # 중요한 조건들을 만족하는지 확인: 학점, 제외 과목, 제외 요일, 그리고 스케줄이 비어있지 않은지
            if credits_ok and excluded_courses_ok and excluded_days_ok and current_schedule:
                schedule_key = frozenset([c.code for c in current_schedule]) 
                
                if schedule_key not in found_schedules_map or found_schedules_map[schedule_key][0] < final_reward:
                    found_schedules_map[schedule_key] = (final_reward, current_schedule)
                
                logging.debug(f"DEBUG: 유효한 시간표 발견! 학점: {final_credits}, 적합도: {final_reward:.2f}")
            else:
                logging.debug(f"DEBUG: 유효하지 않은 시간표: 학점:{final_credits} ({credits_ok}), 제외과목:{excluded_courses_ok}, 제외요일:{excluded_days_ok}, 스케줄비었음:{not current_schedule}")

            self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

            if (episode + 1) % 1000 == 0: 
                current_best_fitness = max(s[0] for s in found_schedules_map.values()) if found_schedules_map else 0.0
                logging.info(f"INFO: 에피소드 {episode + 1}/{num_episodes}, 현재 최고 적합도: {current_best_fitness:.2f}, 엡실론: {self.epsilon:.4f}, Q-테이블 크기: {len(self.q_table)}")

        sorted_schedules = sorted(found_schedules_map.values(), key=lambda x: x[0], reverse=True)
        
        logging.info(f"INFO: 시간표 추천 완료. 최종 {min(num_recommendations, len(sorted_schedules))}개 추천 스케줄 반환.")
        logging.info(f"INFO: 최종 Q-테이블 크기: {len(self.q_table)}")

        return [s[1] for s in sorted_schedules[:num_recommendations]]
