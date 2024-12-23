from .plot_functions import *
from sklearn.metrics import confusion_matrix
import numpy as np
from .dataset_processors import generators_to_dataset


def full_eval(model_ds_dir, history, model, dataset_name, test_generators):
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

    # Generate the PR curve
    plot_pr_curve(model.name, true_labels, predicted_probs,
                  model_ds_dir, dataset_name)

    # Generate the ROC curve
    optimal_threshold = generate_roc_curve(
        model.name, true_labels, predicted_probs, model_ds_dir, dataset_name)

    # Plot test images with optimal threshold
    plot_test_images(test_dataset, model_ds_dir, dataset_name,
                     model, optimal_threshold, optimal=True)

    # Generate predictions with optimal threshold
    predicted_labels_optimal = (
        predicted_probs >= optimal_threshold).astype(int).flatten()

    # Generate the confusion matrix using the optimal threshold
    cm_optimal = confusion_matrix(true_labels, predicted_labels_optimal)
    plot_confusion_matrix(cm_optimal, model_ds_dir,
                          model.name, dataset_name, optimal=True)

    return optimal_threshold


def get_labels_and_predictions(generators, model):
    true_labels = []
    predictions = []
    probabilities = []
    for generator in generators:
        for _ in range(len(generator)):
            X, y = next(generator)
            preds = model.predict(X)
            probabilities.extend(preds.flatten())
            preds = (preds >= 0.5).astype(np.float32).flatten()
            true_labels.extend(y)
            predictions.extend(preds)
    return np.array(true_labels), np.array(predictions), np.array(probabilities)


def extract_evaluation_data(data):
    rows = []
    for model_name, datasets in data.items():
        for dataset_name, metrics in datasets.items():
            row = {
                "Model": model_name,
                "Dataset": dataset_name,
                "Train Size": metrics.get("train_dataset_size"),
                "Val Size": metrics.get("val_dataset_size"),
                "Evaluation Accuracy": metrics["evaluation"]["accuracy"] if "evaluation" in metrics else None,
                "Evaluation Loss": metrics["evaluation"]["loss"] if "evaluation" in metrics else None,
                "Evaluation AUC": metrics["evaluation"]["auc"] if "evaluation" in metrics else None,
                "Evaluation Precision": metrics["evaluation"]["precision"] if "evaluation" in metrics else None,
                "Evaluation Recall": metrics["evaluation"]["recall"] if "evaluation" in metrics else None,
                "Evaluation F1 Score": metrics["evaluation"]["f1_score"] if "evaluation" in metrics else None,
                "Training Time": metrics.get("training_time"),
                "Optimal Threshold": metrics.get("optimal_threshold")
            }
            rows.append(row)
    return rows


def plot_metric_chart(df, metric, dir):
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

    # Customize the plot
    plot.set_title(f"Model Performance Comparison ({metric})")
    plot.set_ylabel(metric)
    plot.set_xlabel("Model")
    plt.xticks(rotation=45)
    plt.legend(title="Dataset")
    plt.tight_layout()

    plt.savefig(os.path.join(dir, f"performance_comparison_{metric}.png"))


def plot_time_extrapolation(df, dir):
    average_training_time = df['Training Time'].mean()
    print(f"Average Training Time: {average_training_time}")
    num_distinct_models = df['Model'].nunique()
    print(f"Number of Distinct Models: {num_distinct_models}")
    num_singular_datasets = df['Dataset'].apply(lambda x: '_' not in x).sum()
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
