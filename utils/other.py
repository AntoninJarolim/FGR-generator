import time

class TimedList(list):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.last_time = time.time()

    def append(self, item):
        assert isinstance(item, dict), "Usage of TimedList is only for dicts"
        current_time = time.time()
        elapsed = current_time - self.last_time
        item['generation_time'] = elapsed
        self.last_time = current_time
        super().append(item)
