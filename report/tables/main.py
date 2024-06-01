import json
from typing import Dict

import pandas as pd

from report.log_parser import LogParser
from report.tables.highlight_rule import get_highlight_rules
from report.tables.saver import LatexSaver
from report.tables.styler import TableStyler
from report.tables.table_argument_parser import TableArgumentParser


def get_data_from_config(json_data: Dict) -> pd.DataFrame:
    data = []
    parser = LogParser()
    for log_name, log_path in json_data['logs'].items():
        log_data_dict = parser.parse(log_path)
        log_data_dict = {'log_name': log_name, **log_data_dict}
        data.append(log_data_dict)
    return data


def main() -> None:
    args = TableArgumentParser(underscores_to_dashes=True).parse_args()
    with args.config.open() as file:
        json_data = json.load(file)

    args.save_path.mkdir(parents=True, exist_ok=True)
    saver = LatexSaver(args.save_path / json_data['name'])

    data = get_data_from_config(json_data)
    data_frame = pd.DataFrame(data)

    rules = get_highlight_rules()
    styler = TableStyler(data_frame, json_data['columns'], rules).style()
    saver.save(styler)



if __name__ == '__main__':
    main()