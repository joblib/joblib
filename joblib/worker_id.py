import os


def worker_id():
    var_value = os.environ.get('JOBLIB_WORKER_ID', None)
    return int(var_value)
