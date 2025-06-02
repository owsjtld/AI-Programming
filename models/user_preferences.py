# models/user_preferences.py
import json # from_dict 메서드에서 json을 사용하지 않지만, 향후 확장성을 위해 남겨둠

class UserPreferences:
    def __init__(self,
                 min_credits: int,
                 max_credits: int,
                 preferred_days: set[int],
                 excluded_courses: list[str],
                 preferred_rating: float,
                 excluded_days: set[int] = None, # set 타입 힌트 사용
                 gap_preference_level: str = 'medium',
                 preferred_grade: int = 0 # preferred_grade 필드 추가 및 기본값 0 설정
                ):
        self.min_credits = min_credits
        self.max_credits = max_credits
        # preferred_days와 excluded_days는 set()으로 받는 것이 더 효율적입니다.
        self.preferred_days = preferred_days if preferred_days is not None else set()
        self.excluded_days = excluded_days if excluded_days is not None else set()
        
        self.excluded_courses = excluded_courses if excluded_courses is not None else []
        self.preferred_rating = preferred_rating
        
        self.gap_preference_level = gap_preference_level
        self.preferred_grade = preferred_grade # preferred_grade 초기화

    def __repr__(self):
        return (f"UserPreferences(min_credits={self.min_credits}, max_credits={self.max_credits}, "
                f"preferred_days={self.preferred_days}, excluded_courses={self.excluded_courses}, "
                f"preferred_rating={self.preferred_rating}, excluded_days={self.excluded_days}, "
                f"gap_preference_level={self.gap_preference_level}, "
                f"preferred_grade={self.preferred_grade})") # preferred_grade repr에 포함

    # --- from_dict 메서드 추가 ---
    @classmethod
    def from_dict(cls, data: dict):
        return cls(
            min_credits=data.get('min_credits', 12),
            max_credits=data.get('max_credits', 18),
            preferred_days=set(data.get('preferred_days', [])),
            excluded_days=set(data.get('excluded_days', [])),
            excluded_courses=data.get('excluded_courses', []),
            preferred_rating=data.get('preferred_rating', 3.0),
            gap_preference_level=data.get('gap_preference_level', 'medium'),
            preferred_grade=data.get('preferred_grade', 0) # preferred_grade 파싱 추가 (없으면 기본값 0)
        )

    # --- to_dict 메서드 추가 (main.py에서 사용될 수 있음) ---
    def to_dict(self):
        return {
            'min_credits': self.min_credits,
            'max_credits': self.max_credits,
            'preferred_days': list(self.preferred_days), # set을 list로 변환
            'excluded_days': list(self.excluded_days),   # set을 list로 변환
            'excluded_courses': self.excluded_courses,
            'preferred_rating': self.preferred_rating,
            'gap_preference_level': self.gap_preference_level,
            'preferred_grade': self.preferred_grade # preferred_grade 포함
        }