from pathlib import Path

from tap import Tap

from lib.data import get_experiment_data
from lib.experiment import run_experiment


class ArgumentParser(Tap):
    path: Path = Path('experiments')
    '''Path to save results'''

    experiment_id: int = 1
    '''
    Experiment id:
        1) 2 normal distributions

        2) 3 normal distributions
    '''


def main():
    args = ArgumentParser().parse_args()
    experiments_path = args.path

    minkowski_parameter = [0.2, 0.6, 1, 1.5, 2, 3, 5]
    T_parameter = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    repeats = 10
    n_points = [10, 100]

    dimension = 20
    n_clusters, prob, mu_list, cov_matrices = get_experiment_data(
        experiment_id=args.experiment_id, dimension=dimension)

    for points in n_points:
        experiment_name = f'Clusters:{n_clusters}, points:{points}'
        output_path = experiments_path / f'exp_1_{points}'

        run_experiment(
            n_clusters=n_clusters,
            distance_parameters=T_parameter,
            minkowski_parameters=minkowski_parameter,
            repeats=repeats,
            n_points=points,
            cov_matrices=cov_matrices,
            prob=prob,
            mu_list=mu_list,
            experiment_name=experiment_name,
            output_path=output_path
        )


if __name__ == '__main__':
    main()
