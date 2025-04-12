from pathlib import Path

from report.argument_parser import ArgumentParser


class TableArgumentParser(ArgumentParser):
    save_path: Path = Path("./data/tables")
