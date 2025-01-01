# Project Mjölnir

## Overview

Project Mjölnir is an AutoML pipeline designed to automatically train and test datasets using a systematic method akin to the brute force method used for password cracking. This pipeline tests out user-defined models on various combinations of all provided datasets. Every possible combination is used to train every model, and each possible pairing is evaluated. The output of the pipeline is a set of aggregated performance comparison charts that are generated to help in determining the best model-combination pairing.

Note that the pipeline is built for binary image classification tasks, but can be extended to other types of tasks with further modifications.

This project was made as part of my dissertation for my MSc in Computing and IT Management at Cardiff University.

## Features

- **Custom Configurations**: Custom configurations for data ingestion, hyperparameters, and model selection are supported.
- **Seamless Preprocessing**: Automatically preprocesses data before training.
- **Automated Training and Evaluation**: Automatically trains and evaluates all possible model-data pairings.
- **Saves Model Artifacts**: Saves every pairing's model artifacts for future use.
- **Comprehensive Data Visualization**: Generates insightful visualizations to help compare model-data pairing performances.

## Important Note

This project has been written in Python 3.12.8. The pipeline will not work on Python 3.11 or lower due to the use of multi-line F-strings. Please ensure you have Python 3.12 or later installed on your system.

## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/selmantabet/mjolnir-automl.git
   ```
2. **Navigate to the project directory**:
   ```bash
   cd mjolnir-automl
   ```
3. **(Optional) - Set up a virtual environment**:

   You may opt to use a virtual environment as follows:

   On Windows

   ```bash
   python -m venv venv
   .\venv\Scripts\activate
   ```

   On UNIX-based systems

   ```bash
   python -m venv venv
   source venv/bin/activate
   ```

4. **Install the required packages using pip**:

   ```bash
   pip install -r requirements.txt
   ```

## Directory Structure

The directory tree should look something like this:

```
mjolnir-automl/
├── data/
├── utils/
│   ├── cfg_validator.py
│   ├── dataset_processors.py
│   ├── evaluator.py
│   ├── img_processing.py
│   ├── initializer.py
│   └── plot_functions.py
├── README.md
├── project-mjolnir.ipynb
├── mjolnir.py
├── cmd_cfg.py
└── requirements.txt
```

## Usage

There are two ways to use the pipeline:

### Method 1: Using the Command Line Interface (CLI)

1. **Prepare the data**:

   Place all the datasets in different folders under the `data` directory.

2. **Configure the pipeline**:

   Modify the `cmd_cfg.py` file with your desired configurations. You may define custom models, hyperparameters, preprocessing parameters, metrics, etc. Feel free to make multiple configurations and save them as different files.

3. **Run the pipeline**:

   Run the following command in the terminal:

   ```bash
   python mjolnir.py --from-py-cfg cmd_cfg.py
   ```

   If you have different configuration files, you rerun the pipeline and swap the `cmd_cfg.py` file with your desired configuration file. You may also further automate this by writing a shell script to iterate over many configuration files.

### Method 2: Using the Jupyter Notebook

1. **Prepare the data**:

   Place all the datasets in different folders under the `data` directory.

2. **Configure the pipeline**:

   Open the `project-mjolnir.ipynb` notebook and configure the pipeline by setting the different parameters in the notebook, the notebook is well documented to guide you through the process.

3. **Run the pipeline**:

   Run the cells in the notebook to execute the pipeline.

## Output

All the output files are saved in the `runs` directory. A new folder is created on each full pipeline execution. The output directory tree is as follows:

```bash
runs/
├── run1/
│   ├── model1/
│   │   ├── model1_dataset1/
│   │   │   ├── history_model1_dataset1.csv
│   │   │   ├── model1_dataset1_model.keras
│   │   │   ├── confusion_matrix_model1_dataset1.png
│   │   │   ├── confusion_matrix_model1_dataset1_optimal.png
│   │   │   ├── pr_curve_model1_dataset1.png
│   │   │   ├── roc_curve_model1_dataset1.png
│   │   │   ├── recall_f1_precision_model1_dataset1.png
│   │   │   ├── loss_accuracy_model1_dataset1.png
│   │   │   ├── test_images_model1_dataset1.png
│   │   │   └── test_images_model1_dataset1_optimal.png
│   │   ├── model1_dataset2/
│   │   ├── ...
│   ├── model2/
│   ├── ...
│   ├── evaluations/
│   │   ├── performance_comparison_accuracy.png
│   │   ├── performance_comparison_precision.png
│   │   ├── performance_comparison_recall.png
│   │   ├── performance_comparison_f1_score.png
│   │   ├── performance_comparison_auc.png
│   │   ├── performance_comparison_Training Time.png
│   │   ├── training_time_extrapolation.png
│   │   ├── weighted_sum_of_metrics.png
│   │   ├── unweighted_sum_of_metrics.png
│   │   ├── train_class_counts.png
│   │   ├── val_class_counts.png
│   │   ├── training_data.csv
│   │   ├── train_dataset_sizes.png
│   │   └── val_dataset_sizes.png
│   ├── run_config.json
│   └── training_results.json
├── run2/
├── ...
```

## Contributing

Contributions are welcome! Please open an [issue](https://github.com/selmantabet/mjolnir-automl/issues) or submit a [pull request](https://github.com/selmantabet/mjolnir-automl/pull).

## License

This project is licensed under the MIT License.

## Contact

For any questions or suggestions, please [contact me here](https://www.selman.io/contact).
