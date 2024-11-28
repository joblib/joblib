"""
Test the 'pretty printing' of job status when the number of jobs is known.
"""

from typing import Any
from time import sleep
import re

from joblib.parallel import Parallel as _Parallel
from joblib.parallel import delayed

SLEEP, N_TASKS = 0.01, 1000
PATTERN = re.compile(r"(Done\s+\d+ out of \d+ \|)")

def dummy_task(x: Any) -> Any:
    sleep(SLEEP)
    return x

class Parallel(_Parallel):

    messages: list[str] = []

    def _print(self, msg):
        self.messages.append(msg)

def test_pretty_print():
    executor = Parallel(n_jobs=2, verbose=10000)
    executor([delayed(dummy_task)(i) for i in range(N_TASKS)])
    lens = set()
    for message in executor.messages:
        if s := PATTERN.search(message):
            a, b = s.span()
            lens.add(b - a)
    assert len(set(lens)) == 1