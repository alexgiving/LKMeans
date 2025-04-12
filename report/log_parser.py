import json
import re
from pathlib import Path
from typing import Dict


class LogParser:
    def __init__(self):
        self._json_pattern = re.compile(r"\{.*\}")

    def parse(self, log_path: Path) -> Dict[str, float]:
        with Path(log_path).open(encoding="utf-8") as log_buff:
            log_data = log_buff.read()

        log_data_lines = log_data.split("\n")
        for log_data_line in log_data_lines:
            if self._json_pattern.match(log_data_line):
                log_data = log_data_line.replace("'", '"')

        try:
            data = json.loads(log_data)
        except json.decoder.JSONDecodeError as exc:
            raise ValueError(f'Error while loading file {log_path}') from exc
        return data
