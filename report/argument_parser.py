from pathlib import Path

from tap import Tap


class ArgumentParser(Tap):
    config: Path