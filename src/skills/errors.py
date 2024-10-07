class SkillException(Exception):
    def __init__(self, message):
        self.message = message
        super().__init__(self.message)

class SkillArgException(Exception):
    def __init__(self, message):
        self.message = message
        super().__init__(self.message)