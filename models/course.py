# models/course.py

import json

class TimeSlot:
    def __init__(self, day: int, start: int, end: int):
        self.day = day
        self.start = start
        self.end = end

    @classmethod
    def from_dict(cls, data):
        return cls(data['day'], data['start'], data['end'])

    def to_dict(self):
        return {'day': self.day, 'start': self.start, 'end': self.end}

    def __repr__(self):
        return f"TimeSlot(day={self.day}, start={self.start}, end={self.end})"

    def __eq__(self, other): # <-- 여기를 수정했습니다: other 인자 추가
        if not isinstance(other, TimeSlot):
            return NotImplemented
        return self.day == other.day and self.start == other.start and self.end == other.end

    def __hash__(self):
        return hash((self.day, self.start, self.end))


class Course:
    def __init__(self, code: str, name: str, professor: str, credits: int, 
                 times: list[TimeSlot], rating: float, location: str = "미정", 
                 difficulty: float = 0.5, grade: int = 0):
        self.code = code
        self.name = name
        self.professor = professor
        self.credits = credits
        self.times = times # list of TimeSlot objects
        self.rating = rating
        self.location = location # 'room' 대신 'location' 사용
        self.difficulty = difficulty
        self.grade = grade

    @classmethod
    def from_dict(cls, data):
        times = [TimeSlot.from_dict(ts) for ts in data.get('times', [])]
        return cls(
            code=data['code'],
            name=data['name'],
            professor=data['professor'],
            credits=data['credits'],
            times=times,
            rating=data.get('rating', 0.0),
            location=data.get('location', '미정'), # 'room' 대신 'location' 사용
            difficulty=data.get('difficulty', 0.5),
            grade=data.get('grade', 0)
        )

    def to_dict(self):
        return {
            'code': self.code,
            'name': self.name,
            'professor': self.professor,
            'credits': self.credits,
            'times': [ts.to_dict() for ts in self.times],
            'rating': self.rating,
            'location': self.location, # 'room' 대신 'location' 사용
            'difficulty': self.difficulty,
            'grade': self.grade
        }

    def __repr__(self):
        return (f"Course(code='{self.code}', name='{self.name}', "
                        f"professor='{self.professor}', credits={self.credits}, "
                        f"times={self.times}, rating={self.rating}, "
                        f"location='{self.location}', difficulty={self.difficulty}, "
                        f"grade={self.grade})")

    def __eq__(self, other):
        if not isinstance(other, Course):
            return NotImplemented
        return self.code == other.code

    def __hash__(self):
        return hash(self.code)