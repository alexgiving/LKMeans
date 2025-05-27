import json
from typing import Dict

import pandas as pd

from report.log_parser import LogParser
from report.tables.highlight_rule import get_highlight_rules
from report.tables.saver import LatexSaver
from report.tables.styler import TableStyler
from report.tables.table_argument_parser import TableArgumentParser


def parse_log(parser: LogParser, line: str) -> Dict[str, float]:
    if isinstance(line, dict):
        return line
    if len(line.split(' ')) > 1:
        return json.loads(line.replace('\'', '"'))
    return parser.parse(line)


def main() -> None:
    args = TableArgumentParser(underscores_to_dashes=True).parse_args()
    with args.config.open() as file:
        json_data = json.load(file)

    args.save_path.mkdir(parents=True, exist_ok=True)
    saver = LatexSaver(args.save_path / json_data['name'])

    parser = LogParser()
    data = []

    log_data_dict = parse_log(parser, json_data['baseline'])
    log_data_dict = {'log_name': "Baseline", **log_data_dict}
    data.append(log_data_dict)

    for logs_block in json_data['logs'].values():
        for log_name, log_path in logs_block.items():
            log_data_dict = parse_log(parser, log_path)
            log_data_dict = {'log_name': log_name, **log_data_dict}
            data.append(log_data_dict)
    data_frame = pd.DataFrame(data)

    rules = get_highlight_rules()
    styler = TableStyler(data_frame, json_data['columns'], rules).style()
    saver.save(styler)



if __name__ == '__main__':
    main()
