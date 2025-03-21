import json
from collections import defaultdict
from typing import Any, Dict, List

from matplotlib import pyplot as plt
from matplotlib.axes import Axes

from report.charts.chart_argument_parser import ChartArgumentParser
from report.log_parser import LogParser
from report.metric_name_processor import process_metric_name


def select_metric(all_data: Dict[str, List[Dict[str, float]]], metric: str) -> Dict[str, List[float]]:
    data = {}
    for block_name, block_metrics in all_data.items():
        data[block_name] = [metrics[metric] for metrics in block_metrics]
    return data


def configure_x_axis(axes: Axes, json_data: Dict[str, Any]) -> Axes:
    axes.set_xticks(ticks=range(len(json_data['xticks'])))
    axes.set_xticklabels(json_data['xticks'])

    xlabel = json_data.get('xlabel')
    if xlabel is not None:
        axes.set_xlabel(xlabel=xlabel)
    return axes


def parse_log(parser: LogParser, line: str) -> Dict[str, float]:
    if len(line.split(' ')) > 1:
        return json.loads(line.replace('\'', '"'))
    return parser.parse(line)


def main() -> None:
    args = ChartArgumentParser(underscores_to_dashes=True).parse_args()
    with args.config.open() as file:
        json_data = json.load(file)

    args.save_path.mkdir(parents=True, exist_ok=True)

    parser = LogParser()
    data = defaultdict(list)
    for block_name, logs_block in json_data['logs'].items():
        for log_path in logs_block.values():
            log_data_dict = parse_log(parser, log_path)
            data[block_name].append(log_data_dict)

    baseline_data_dict = None
    baseline_path = json_data.get('baseline', None)
    if baseline_path:
        baseline_data_dict = parse_log(parser, baseline_path)

    for metric in json_data['plot_metrics']:
        config_name = json_data['name']
        chart_name = args.save_path / f'{config_name}_{metric}.png'
        prepared_data = select_metric(data, metric)

        figure = plt.figure(figsize=(5,4), dpi=800)
        axes = figure.gca()

        num_measurements_in_line = 0
        for line_name, values in prepared_data.items():
            axes.plot(values, label=line_name)
            num_measurements_in_line = max(num_measurements_in_line, len(values))

        if baseline_data_dict is not None:
            baseline_values = [baseline_data_dict[metric]]*num_measurements_in_line
            axes.plot(baseline_values, '--', label="Baseline")

        axes.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left',
                      ncols=2, mode="expand", borderaxespad=0.)

        axes = configure_x_axis(axes, json_data)

        metric_name = process_metric_name(metric)
        if args.metric_to_title:
            axes.set_title(metric_name)
        else:
            axes.set_ylabel(ylabel=metric_name)

        figure.tight_layout()
        figure.savefig(chart_name)
        plt.close(figure)


if __name__ == '__main__':
    main()
