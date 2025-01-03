# Experiments Reporting System

A tool for generating charts and tables from experimental logs with customizable configuration.

## Features
- Generate detailed charts for visualizing experimental metrics.
- Create comprehensive tables summarizing key performance indicators.
- Easily configurable through a JSON file.

## Installation
Ensure you have Python installed and required dependencies. You can install the necessary packages with:
```bash
pip install -r requirements.txt
```

## Usage

### Generate chart
To create a chart based on the logs, use the following command:
```bash
python report/tables/main.py --config logs/config.json
```

### Generate table
To generate a table summarizing the results:
```bash
python report/tables/main.py --config logs/config.json
```

### Example of configuration file
Below is an example of a configuration file (config.json) to customize the behavior of the system:
```json
{
    "logs": {
        "log_block": {
            "name of log 1": "path_to_log_1.log",
            "name of log 2": "path_to_log_2.log"
        }
    },
    "name": "prefix_of_output_file",
    "columns": [
        "log_name",
        "ari",
        "ami",
        "completeness",
        "homogeneity",
        "nmi",
        "v_measure",
        "accuracy",
        "time",
        "inertia"
    ],
    "plot_metrics": [
        "ari",
        "ami",
        "time",
    ]
}
```

### Explanation of Configuration Fields:
- logs: Specifies the log files to process.
- log_block: Contains the log names and their respective file paths.
- name: Prefix for the output files (e.g., charts and tables).
- columns: Defines the metrics and details to include in the table.
- plot_metrics: Specifies the metrics to visualize in the chart.

## Example Output
The system will generate:
1.	Charts visualizing selected metrics (e.g., ARI, AMI, Time).
2.	Tables summarizing performance metrics like accuracy, homogeneity, and inertia.

## Contributing
Feel free to open issues or submit pull requests for enhancements.
