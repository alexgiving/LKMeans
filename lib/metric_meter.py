from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class MetricTable:
    def __init__(self) -> None:
        self.frames = []

    def add_frame(self, frame: pd.DataFrame) -> None:
        self.frames.append(frame)

    def add_empty_frame(self, time: bool) -> None:
        empty = 'N/A'
        data = {'ARI': empty, 'AMI': empty}
        if time:
            data['Time'] = empty
        frame = pd.DataFrame(data, [empty])
        self.frames.append(frame)

    def get_table(self) -> pd.DataFrame:
        return pd.concat(self.frames, join="inner")

    def get_latex_table(self, caption: str = '') -> str:
        table = self.get_table()
        return table.to_latex(index=True, escape=True, caption=caption)


def insert_hline(latex_str: str) -> str:
    lines_strings = latex_str.splitlines()
    result = []

    for line in lines_strings:
        if 'N/A' in line:
            result.append('\\midrule')
        else:
            result.append(line)
    result = '\n'.join(result)
    return result


class MetricMeter:
    def __init__(self) -> None:
        self.ari = []
        self.ami = []
        self.inertia = []
        self.time = []

    def add_ari(self, value: float) -> None:
        self.ari.append(value)

    def add_ami(self, value: float) -> None:
        self.ami.append(value)

    def add_inertia(self, value: float) -> None:
        self.inertia.append(value)

    def add_time(self, value: float) -> None:
        self.time.append(value)

    def get_average(self) -> Tuple[float, float, float, float]:
        return float(np.mean(self.ari)), float(np.mean(self.ami)), float(np.mean(self.inertia)), float(np.mean(self.time))


class GraphicMeter(MetricMeter):
    def __init__(self, p: List) -> None:
        super().__init__()
        self.p = p

    def get_graph(self, key: str):
        values = {'ARI': self.ari, 'AMI': self.ami,
                  'Inertia': self.inertia, 'Time': self.time}

        fig, ax = plt.subplots()
        ax.set_xlabel('p')

        ax.plot(self.p, values[key], '-o')
        ax.set_ylabel(key)
        ax.set_title(f'{key} vs. p')
        return fig
