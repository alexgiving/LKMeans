from pathlib import Path

from report.argument_parser import ArgumentParser


class ChartArgumentParser(ArgumentParser):
    save_path: Path = Path("./data/charts")
    metric_to_title: bool = False
