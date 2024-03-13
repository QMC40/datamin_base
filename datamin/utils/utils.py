from typing import Dict, Tuple, Union

import numpy as np


def add_to_dict(a: Dict, b: Dict) -> None:
    for k in a:
        a[k] += b[k]


def itemize_dict(d: Dict) -> Dict:
    return {k: v.item() for k, v in d.items()}


class TempScheduler:
    def __init__(self, start_temp: float, end_temp: float, n_steps: int):
        self.start_temp = start_temp
        self.end_temp = end_temp
        self.n_steps = n_steps
        self.temp = start_temp
        self.temp_step = (self.start_temp - self.end_temp) / float(self.n_steps)

    def step(self) -> None:
        self.temp = np.clip(self.temp - self.temp_step, self.end_temp, self.start_temp)

    def get_temp(self) -> float:
        return self.temp


class Stat:
    def __init__(self) -> None:
        self.tot = 0.0
        self.n = 0.0

    def __iadd__(self, x: Union["Stat", Tuple[float, float], float]) -> "Stat":
        if isinstance(x, Stat):
            self.tot += x.tot
            self.n += x.n
            assert False
        elif isinstance(x, tuple):
            value, k = x
            self.tot += value * k
            self.n += k
        else:
            self.tot += x
            self.n += 1
        return self

    def avg(self) -> float:
        return self.tot / self.n
