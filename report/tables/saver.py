from abc import ABC, abstractmethod
from pathlib import Path

from pandas.io.formats.style import Styler


class Saver(ABC):
    def __init__(self, file_name: Path) -> None:
        self._file_name = file_name
        self._convert_path()

    @abstractmethod
    def _convert_path(self) -> None:
        ...

    @abstractmethod
    def save(self, styler: Styler) -> None:
        ...


class LatexSaver(Saver):

    def _convert_path(self) -> None:
        self._file_name = self._file_name.with_suffix('.tex')

    def save(self, styler: Styler) -> None:
        with self._file_name.open('w') as file:
            styler.format(escape='latex', precision=2).to_latex(file, convert_css=True)
