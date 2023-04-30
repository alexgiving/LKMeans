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
