from functools import wraps
from time import time
from datetime import datetime

def _prnt(s):
    print(s)

def measure(method):
    @wraps(method)
    def wrapper(*args, **kwargs):
        mname = method.__name__
        s = time()
        _prnt(f"Start  {mname} at: {datetime.now()}")
        res = method(*args, **kwargs)
        e = time()
        mins, secs = divmod(e - s, 60)
        _prnt(f"Finish {mname} at: {datetime.now()}")
        _prnt(f"Total time of {mname}: {mins:02.0f}:{secs:05.2f}s")

        return res

    return wrapper
