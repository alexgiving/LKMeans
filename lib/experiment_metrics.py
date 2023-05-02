from typing import List, Optional

import numpy as np
import pandas as pd


def get_average_experiment_metrics(
        ari: float,
        ami: float,
        inertia: float,
        name: Optional[str] = 'Experiment',
        time: Optional[float] = None) -> pd.DataFrame:

    data = {'ARI': f'{ari:.2f}', 'AMI': f'{ami:.2f}',
            'Inertia': f'{inertia:.2f}'}
    if time:
        data['Time'] = f'{time:.2f}'
    frame = pd.DataFrame(data, [name])
    return frame
