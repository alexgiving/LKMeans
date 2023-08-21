from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class MetricTable:
    def __init__(self) -> None:
        self.frames = []

    def add_frame(self, frame: pd.DataFrame) -> None:
        self.frames.append(frame)

    # pylint: disable=too-many-arguments
    def add_to_frame(self,
                     ari: float,
                     ami: float,
                     inertia: float,
                     time: float,
                     name: Optional[str] = 'Experiment'
                     ) -> None:
        data = {'ARI': f'{ari:.2f}', 'AMI': f'{ami:.2f}',
                'Inertia': f'{inertia:.2f}', 'Time': f'{time:.2f}'}
        frame = pd.DataFrame(data, [name])
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

    def add_combination(self, ari: float, ami: float, inertia: float, time: float) -> None:
        self.add_ari(abs(ari))
        self.add_ami(abs(ami))
        self.add_inertia(inertia)
        self.add_time(time)

    def get_average(self) -> tuple[float, float, float, float]:
        return float(np.mean(self.ari)), float(np.mean(self.ami)), \
            float(np.mean(self.inertia)), float(np.mean(self.time))


class GraphicMeter(MetricMeter):
    def __init__(self, base: list, base_name: str) -> None:
        super().__init__()
        self.base = base
        self.base_name = base_name

    def get_graph(self, key: str):
        values = {'ARI': self.ari, 'AMI': self.ami,
                  'Inertia': self.inertia, 'Time': self.time}

        fig, ax = plt.subplots(figsize=(5, 4))
        param = values[key]
        ax.plot(self.base, param, '-o')
        ax.grid(True, color='gray', linestyle='--', linewidth=0.5)

        if self.base_name == 'p':
            ax.set_xticks(self.base)
        else:
            ax.set_xticks(np.linspace(0, 1, 11))
        ax.set_xlabel(self.base_name)

        if key in ('ARI', 'AMI'):
            ax.set_yticks(np.arange(0, 1.1, 0.1))
        else:
            ax.set_yticks(np.linspace(np.min(param), np.max(param), 10))
        ax.set_ylabel(key)
        # ax.set_title(f'{key} vs. {self.base_name}')
        return fig
