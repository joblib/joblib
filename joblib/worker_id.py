import os


def worker_id():
    wid = os.environ.get('JOBLIB_WORKER_ID', None)
    if wid is None:
        return -1
    return int(wid)
