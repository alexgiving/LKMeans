import json
from collections import defaultdict
from typing import Dict, List

import numpy as np
from matplotlib import pyplot as plt

from report.charts.chart_argument_parser import ChartArgumentParser
from report.log_parser import LogParser
from report.metric_name_processor import process_metric_name


def select_metric(all_data: Dict[str, List[Dict[str, float]]], metric: str) -> Dict[str, List[float]]:
    data = {}
    for block_name, block_metrics in all_data.items():
        data[block_name] = [metrics[metric] for metrics in block_metrics]
    return data


def main() -> None:
    args = ChartArgumentParser(underscores_to_dashes=True).parse_args()
    with args.config.open() as file:
        json_data = json.load(file)

    args.save_path.mkdir(parents=True, exist_ok=True)

    parser = LogParser()
    data = defaultdict(list)
    for block_name, logs_block in json_data['logs'].items():
        if block_name == 'LKMeans':
            continue
        for log_path in logs_block.values():
            if len(log_path.split(' ')) > 1:
                log_data_dict = json.loads(log_path.replace('\'', '"'))
            else:
                log_data_dict = parser.parse(log_path)
            data[block_name].append(log_data_dict)

    for metric in json_data['plot_metrics']:
        config_name = json_data['name']
        chart_name = args.save_path / f'{config_name}_{metric}.png'
        prepared_data = select_metric(data, metric)
        figure = plt.figure(figsize=(4,4), dpi=800)
        axes = figure.gca()
        for line_name, values in prepared_data.items():
            axes.plot(values, label=line_name)
        axes.legend()
        axes.set_xticks(ticks=np.linspace(0, 3, 4))
        axes.set_xticklabels(['0', '0.1', '0.15', '0.2'])
        axes.set_title(process_metric_name(metric))
        figure.tight_layout()
        figure.savefig(chart_name)
        plt.close(figure)


if __name__ == '__main__':
    main()
