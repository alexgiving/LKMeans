import json
from pathlib import Path
from typing import Dict


class LogParser:
    def parse(self, log_path: Path) -> Dict[str, float]:
        with Path(log_path).open(encoding='utf-8') as log_buff:
            log_data = log_buff.read()
        log_data = log_data.replace('\'', '"')
        return json.loads(log_data)
