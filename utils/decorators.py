import time


def measure_time(function):
    def wrapper(*args, **kwargs):
        start = time.time()
        ret = function(*args, **kwargs)
        end = time.time()
        print(function.__name__, 'took ', end - start, 'seconds')
        return ret

    return wrapper
