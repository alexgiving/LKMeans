from typing import Dict


def process_metric_name(metric: str) -> str:
    name_map: Dict[str, str] = {
        'ari': 'ARI',
        'ami': 'AMI',
        'nmi': 'NMI',
        'v_measure': 'V-measure'
    }
    return name_map[metric] if metric in name_map else metric.capitalize()
