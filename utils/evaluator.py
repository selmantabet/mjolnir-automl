""" 
The Evaluation Module - Part of the `utils` library for Project Mjölnir

Developed by Selman Tabet @ https://selman.io/
----------------------------------------------
This module contains functions for evaluating models, aggregating results and plotting evaluation metrics.
"""
from .plot_functions import *
from sklearn.metrics import confusion_matrix
import numpy as np
from .dataset_processors import generators_to_dataset
import json
from .initializer import METRIC_TITLES


def full_evaluation(model_ds_dir, history, model, dataset_name, test_generators):
    """
    Perform a full evaluation of a trained model on a given dataset.

    Arguments:
    -----
        model_ds_dir (`str`): Directory where the model and evaluation results will be saved.
        history (`history.History`): Training history object containing metrics and loss values.
        model (`tf.keras.Model`): Trained model to be evaluated.
        dataset_name (`str`): Name of the dataset being evaluated.
        test_generators (`list[ImageDataGenerator]`): List of test data generators.

    Returns:
    -----
        `float`: Optimal threshold value for binary classification based on the Precision-Recall curve.

    This function performs the following steps:
        - Plots training history metrics and saves them.
        - Retrieves true labels and predicted labels/probabilities from the test generators.
        - Plots test images with predictions.
        - Generates and plots a confusion matrix using the default threshold of `0.5`.
        - Generates and plots the Receiver Operating Characteristic (ROC) curve.
        - Generates and plots the Precision-Recall (PR) curve and finds the optimal threshold.
        - Plots test images with predictions using the optimal threshold.
        - Generates and plots a confusion matrix using the optimal threshold.

    """

    print(f"Evaluating {model.name} on {dataset_name}...")
    # Plot metrics and save the history
    plot_history(model_ds_dir, history, model.name, dataset_name)
    # Plot the test images with predictions

    true_labels, predicted_labels, predicted_probs = get_labels_and_predictions(
        test_generators, model)

    test_dataset = generators_to_dataset(test_generators)
    plot_test_images(test_dataset, model_ds_dir, dataset_name, model)

    # Generate the confusion matrix using the default threshold of 0.5
    cm = confusion_matrix(true_labels, predicted_labels)
    plot_confusion_matrix(cm, model_ds_dir, model.name, dataset_name)

    # Generate the ROC curve
    plot_roc_curve(model.name, true_labels, predicted_probs,
                   model_ds_dir, dataset_name)

    # Generate the PR curve and find the optimal threshold
    optimal_threshold = plot_pr_curve(model.name, true_labels, predicted_probs,
                                      model_ds_dir, dataset_name)

    # Plot test images with optimal threshold
    plot_test_images(test_dataset, model_ds_dir, dataset_name,
                     model, optimal_threshold, optimal=True)

    # Generate predictions with optimal threshold
    predicted_labels_optimal = (
        predicted_probs >= optimal_threshold).astype(np.float32).flatten()

    # Generate the confusion matrix using the optimal threshold
    cm_optimal = confusion_matrix(true_labels, predicted_labels_optimal)
    plot_confusion_matrix(cm_optimal, model_ds_dir,
                          model.name, dataset_name, optimal=True)

    return optimal_threshold


def get_labels_and_predictions(generators, model, threshold=0.5):
    """
    Generate true labels, predictions, and prediction probabilities from data generators and a model.

    Arguments:
    -----
        generators (`list[ImageDataGenerator]`): A list of data generators that yield batches of data (X, y).
        model (`keras.Model`): A trained Keras model used to make predictions.
        threshold (`float`, optional): The threshold for converting probabilities to binary predictions. Defaults to `0.5`.

    Returns:
    -----
     `tuple`: A tuple containing three numpy arrays:
        - true_labels (`np.ndarray`): The true labels from the data generators.
        - predictions (`np.ndarray`): The binary predictions made by the model.
        - probabilities (`np.ndarray`): The raw prediction probabilities from the model.
    """

    true_labels = []
    predictions = []
    probabilities = []
    for generator in generators:
        for _ in range(len(generator)):
            X, y = next(generator)
            preds = model.predict(X)
            probabilities.extend(preds.flatten())
            preds = (preds >= threshold).astype(np.float32).flatten()
            true_labels.extend(y)
            predictions.extend(preds)
    return np.array(true_labels), np.array(predictions), np.array(probabilities)


def plot_sum_of_metrics_heatmaps(eval_dir, df, metric_weights=None):
    """
    Plots heatmaps for the weighted and unweighted sum of metrics.
    This function takes a `pandas.DataFrame` containing evaluation metrics, min-max normalizes
    the metrics, calculates the weighted and unweighted sum of metrics for 
    each Dataset-Model pair, and plots heatmaps for both.

    Arguments:
    -----
        eval_dir (`str`): Directory where the heatmap images will be saved.
        df (`pd.DataFrame`): `pandas.DataFrame` containing the evaluation metrics.
        metric_weights (`dict`, optional): Dictionary containing weights for each metric. Keys should be metric names in lowercase. If `None`, no weights are applied.

    Returns:
    -----
        `None` - but saves the heatmaps to the specified directory.
    """

    # Drop irrelevant columns
    df = df.drop(columns=['Train Size', 'Val Size', 'Training Time',
                 'Optimal Threshold', 'Train Counts', 'Val Counts'])
    if metric_weights is not None or metric_weights != {}:
        # Ensure metric_weights keys are lowercase
        metric_weights = {k.lower(): v for k, v in metric_weights.items()}

        # Normalize each metric in the dataframe using min-max normalization
        for metric in metric_weights.keys():
            if metric in df.columns.str.lower():
                df[metric] = (df[metric] - df[metric].min()) / \
                    (df[metric].max() - df[metric].min())

        # Group by Dataset-Model pairs
        grouped = df.groupby(['Dataset', 'Model'])
        # Initialize an empty DataFrame to store the results
        results = []
        for (dataset, model), group in grouped:
            weighted_sum = 0
            for metric, weight in metric_weights.items():
                if metric in group.columns.str.lower():
                    weighted_sum += group[metric].sum() * weight
                else:
                    print(
                        f"Metric '{metric}' not found in DataFrame, skipping...")
            results.append({
                'Dataset': dataset,
                'Model': model,
                'Weighted Sum': weighted_sum
            })

        # Convert results to DataFrame
        results_df = pd.DataFrame(results)

        # Plot heatmap
        plt.figure(figsize=(23, 14))
        heatmap_data = results_df.pivot(
            index='Dataset', columns='Model', values='Weighted Sum')
        sns.heatmap(heatmap_data, annot=True, fmt=".2f", cmap='RdYlGn')
        plt.title('Weighted Sum of Metrics Heatmap',
                  fontsize=26, fontweight='bold')
        plt.savefig(os.path.join(
            eval_dir, "weighted_sum_of_metrics_heatmap.png"))

    # Calculate the unweighted sum of metrics
    for metric in df.columns:
        # Normalization and assuming every other column is a metric
        if metric.lower() not in ['dataset', 'model']:
            df[metric] = (df[metric] - df[metric].min()) / \
                (df[metric].max() - df[metric].min())

    # Group by 'Dataset' and 'Model'
    grouped = df.groupby(['Dataset', 'Model'])

    # Initialize an empty DataFrame to store the results
    results = []

    for (dataset, model), group in grouped:
        unweighted_sum = group.drop(
            # Sum of all metrics
            columns=['Dataset', 'Model']).sum(axis=1).sum()
        results.append({
            'Dataset': dataset,
            'Model': model,
            'Unweighted Sum': unweighted_sum
        })

    # Convert results to DataFrame
    results_df = pd.DataFrame(results)

    # Plot heatmap
    plt.figure(figsize=(23, 14))
    heatmap_data = results_df.pivot(
        index='Dataset', columns='Model', values='Unweighted Sum')
    sns.heatmap(heatmap_data, annot=True, fmt=".2f", cmap='RdYlGn')
    plt.title('Unweighted Sum of Metrics Heatmap',
              fontsize=26, fontweight='bold')
    plt.savefig(os.path.join(
        eval_dir, "unweighted_sum_of_metrics_heatmap.png"))


def extract_evaluation_data(data):
    """
    Extracts evaluation data from a nested dictionary structure and returns it as a list of dictionaries.

    Arguments:
    -----
        data (`dict`): Training results loaded from `training_results.json`

    Returns:
    -----
    `list`: A list of dictionaries, where each dictionary represents a row of evaluation data 
              with the following keys:
        - "Model": The name of the model.
        - "Dataset": The name of the dataset (or combination).
        - "Train Size": The size of the training dataset.
        - "Val Size": The size of the validation dataset.
        - "Training Time": The time taken to train the model.
        - "Optimal Threshold": The optimal threshold value.
        - "Train Counts": A JSON string of training counts.
        - "Val Counts": A JSON string of validation counts.
        - Additional keys for each evaluation metric present in the data.
    """

    rows = []
    for model_name, datasets in data.items():
        for dataset_name, metrics in datasets.items():
            row = {
                "Model": model_name,
                "Dataset": dataset_name,
                "Train Size": metrics.get("train_dataset_size"),
                "Val Size": metrics.get("val_dataset_size"),
                "Training Time": metrics.get("training_time"),
                "Optimal Threshold": metrics.get("optimal_threshold"),
                "Train Counts": json.dumps(metrics.get("train_counts")),
                "Val Counts": json.dumps(metrics.get("val_counts")),
            }
            evaluation_metrics = metrics.get("evaluation")
            for metric, value in evaluation_metrics.items():
                row[metric] = value
            rows.append(row)
    return rows


def plot_dataset_sizes(df, dir):
    """
    Plots dataset sizes and class counts for training and validation datasets.

    Arguments:
    -----
    df (`pd.DataFrame`): `pandas.DataFrame` containing dataset information. Expected columns:
        - 'Dataset': Name of the dataset
        - 'Train Size': Size of the training dataset
        - 'Val Size': Size of the validation dataset
        - 'Train Counts': JSON string representing class counts in the training dataset
        - 'Val Counts': JSON string representing class counts in the validation dataset

    dir (`str`): Directory path where the plots will be saved.

    Returns:
    -----
        `None`

    The function generates and saves the following plots:
    1. Bar plot of training dataset sizes (`train_dataset_sizes.png`)
    2. Bar plot of validation dataset sizes (`val_dataset_sizes.png`)
    3. Bar plot of training sample sizes per class for each dataset (`train_class_counts.png`)
    4. Bar plot of validation sample sizes per class for each dataset (`val_class_counts.png`)

    The plots are saved in the specified directory.
    """

    # Set plot style
    sns.set_theme(style="whitegrid")

    # Create a bar plot for Dataset Sizes
    plt.figure(figsize=(14, 8))
    plot = sns.barplot(
        data=df,
        x="Dataset",
        y="Train Size",
        palette="viridis",
        hue="Dataset",
        legend=False
    )

    # Customize the plot
    plot.set_title("Dataset Sizes")
    plot.set_ylabel("Train Size")
    plot.set_xlabel("Dataset")
    plt.xticks(rotation=45)
    plt.tight_layout()

    plt.savefig(os.path.join(dir, "train_dataset_sizes.png"))

    plt.figure(figsize=(14, 8))
    plot = sns.barplot(
        data=df,
        x="Dataset",
        y="Val Size",
        palette="viridis",
        hue="Dataset",
        legend=False
    )

    # Customize the plot
    plot.set_title("Validation Dataset Sizes")
    plot.set_ylabel("Val Size")
    plot.set_xlabel("Dataset")
    plt.xticks(rotation=45)
    plt.tight_layout()

    plt.savefig(os.path.join(dir, "val_dataset_sizes.png"))

    # Create two bar plots showing each dataset's class counts for train and val
    train_data = []
    val_data = []

    # Collect train and val counts
    for i, row in df.iterrows():
        dataset_name = row['Dataset']
        train_counts_str = row.get('Train Counts', '')
        train_counts_dict = json.loads(
            train_counts_str) if train_counts_str else {}
        val_counts_str = row.get('Val Counts', '')
        val_counts_dict = json.loads(val_counts_str) if val_counts_str else {}
        for class_name, count in train_counts_dict.items():
            train_data.append(
                {'Dataset': dataset_name, 'Class': class_name, 'Count': count})
        for class_name, count in val_counts_dict.items():
            val_data.append(
                {'Dataset': dataset_name, 'Class': class_name, 'Count': count})

    # Convert to DataFrame
    train_df = pd.DataFrame(train_data)
    val_df = pd.DataFrame(val_data)

    # Plot train counts
    plt.figure(figsize=(12, 6))
    sns.barplot(data=train_df, x='Dataset', y='Count',
                hue='Class', palette='viridis')
    plt.title("Training sample sizes per class for each dataset")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(dir, "train_class_counts.png"))

    # Plot val counts
    plt.figure(figsize=(12, 6))
    sns.barplot(data=val_df, x='Dataset', y='Count',
                hue='Class', palette='viridis')
    plt.title("Validation sample sizes per class for each dataset")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(dir, "val_class_counts.png"))


def plot_metric_chart(df, metric, dir, highlight_max=True):
    """
    Plots a bar chart comparing the specified metric across different models and datasets.

    Arguments:
    -----
        df (`pd.DataFrame`): `pandas.DataFrame` containing the data to plot. It must have columns 'Model', 'Dataset', and the specified metric.
        metric (`str`): The name of the metric to plot.
        dir (`str`): The directory where the plot image will be saved.
        highlight_max (`bool`, optional): If `True`, highlights the bar with the maximum value in red. If `False`, highlights the bar with the minimum value in red. Default is `True`.

    Returns:
    -----
        `None` - The function saves the plot as a PNG file in the specified directory.

    Notes:
    - The plot style is set to `whitegrid` using `seaborn`.
    - The plot is saved with the filename format `performance_comparison_{metric}.png`.
    - The title of the plot is customized based on the metric.
    - The x-axis labels are rotated by 45 degrees for better readability.
    """

    # Set plot style
    sns.set_theme(style="whitegrid")

    # Create a bar plot for Evaluation Accuracy comparison across models and datasets
    plt.figure(figsize=(14, 8))
    plot = sns.barplot(
        data=df,
        x="Model",
        y=metric,
        hue="Dataset",
        palette="viridis"
    )
    if highlight_max:
        # Highlight the highest bar in the entire chart
        max_value = df[metric].max()
        for patch in plot.patches:
            if patch.get_height() == max_value:
                patch.set_edgecolor('red')
                patch.set_linewidth(3)
    else:
        min_value = df[metric].min()
        for patch in plot.patches:
            if patch.get_height() == min_value:
                patch.set_edgecolor('red')
                patch.set_linewidth(3)
    # Customize the plot
    metric_title = METRIC_TITLES.get(metric, metric)
    plot.set_title(f"Model Performance Comparison ({metric_title})")
    plot.set_ylabel(metric)
    plot.set_xlabel("Model")
    plt.xticks(rotation=45)
    plt.figtext(0.99, 0.01, "The highlighted bar with the red outline is the best one",
                horizontalalignment='right', fontsize=10, color='red')
    plt.legend(title="Dataset")
    plt.tight_layout()

    plt.savefig(os.path.join(dir, f"performance_comparison_{metric}.png"))


def plot_time_extrapolation(df, dir):
    """
    Plots a 3D surface extrapolation of training time based on the number of models and datasets.

    Arguments:
    -----
        df (`pd.DataFrame`): `pandas.DataFrame` containing the training data with columns 'Training Time', 'Model', and 'Dataset'.
        dir (`str`): Directory path where the plot image will be saved.

    Returns:
    -----
        `None` - The function saves the plot as a PNG file in the specified directory.

    The function calculates the average training time, the number of distinct models, and the number of singular datasets 
    (datasets without an underscore in their name). It then creates a 3D surface plot showing the extrapolated training 
    time as a function of the number of models and datasets, and saves the plot as `training_time_extrapolation.png` 
    in the specified directory.
    """

    average_training_time = df['Training Time'].mean()
    print(f"Average Training Time: {average_training_time}")
    num_distinct_models = df['Model'].nunique()
    print(f"Number of Distinct Models: {num_distinct_models}")
    num_singular_datasets = df[df['Dataset'].apply(
        lambda x: '_' not in x)]['Dataset'].nunique()
    print(f"Number of Singular Datasets: {num_singular_datasets}")
    # Define the range of models and datasets
    # Number of models from 1 to 10
    models = np.arange(1, num_distinct_models+2, 1)
    # Number of datasets from 1 to 10
    datasets = np.arange(1, num_singular_datasets+2, 1)

    # Create meshgrid for 3D plotting
    models_grid, datasets_grid = np.meshgrid(models, datasets)

    # Calculate training time
    training_time = models_grid * \
        (2**datasets_grid - 1) * average_training_time/60

    # Plot the 3D surface
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(models_grid, datasets_grid, training_time, cmap='viridis')

    # Add labels and title
    ax.set_xlabel('Number of Models')
    ax.set_ylabel('Number of Datasets')
    ax.set_zlabel('Training Time (hours)')
    ax.set_title('Training Time vs. Number of Models vs. Number of Datasets')
    plt.savefig(os.path.join(dir, "training_time_extrapolation.png"))
