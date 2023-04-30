from typing import List, Optional

import numpy as np
import pandas as pd


def get_average_experiment_metrics(
        ari: List,
        ami: List,
        name: Optional[str] = 'Experiment',
        time: Optional[List[float]] = None) -> pd.DataFrame:

    data = {'ARI': f'{np.mean(ari):.2f}', 'AMI': f'{np.mean(ami):.2f}'}
    if time:
        data['Time'] = f'{np.mean(time):.2f}'
    frame = pd.DataFrame(data, [name])
    return frame
